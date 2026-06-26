"""Train DQN agent on Bomberman environment."""

import os
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import cupy as cp
import numpy as np

from bomberman import BombermanEnv
from bomberman.entities import GameConfig
from model.network import DQNetwork
from model.optimizer import AdamWOptimizer
from model.target import TargetEstimator

_MODEL_SAVE_LOCK = threading.Lock()
_MODEL_SAVE_RETRY_DELAYS = (0.05, 0.1, 0.25, 0.5, 1.0)

class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer - samples important transitions more frequently."""
    
    def __init__(self, capacity: int = 10000, alpha: float = 0.6):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha  # Priority exponent (0 = uniform, 1 = full prioritization)
        self.epsilon = 1e-6  # Small constant to ensure non-zero priorities
    
    def push(self, state, action, reward, next_state, done, priority=None):
        """Store experience with max priority (new experiences are important)."""
        max_priority = max(self.priorities) if self.priorities else 1.0
        if priority is None:
            priority = max_priority
        
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(priority)
    
    def sample(self, batch_size: int, beta: float = 0.4):
        """Sample batch with prioritized sampling and importance sampling weights."""
        if len(self.buffer) == 0:
            raise ValueError("Replay buffer is empty")

        priorities = np.array(self.priorities, dtype=np.float64)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities, replace=False)

        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        batch = [self.buffer[index] for index in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            cp.array(states, dtype=cp.float32),
            cp.array(actions, dtype=cp.int32),
            cp.array(rewards, dtype=cp.float32),
            cp.array(next_states, dtype=cp.float32),
            cp.array(dones, dtype=cp.float32),
            cp.array(weights, dtype=cp.float32),
            indices,
        )
    
    def update_priorities(self, indices, td_errors):
        """Update priorities based on new TD errors."""
        for index, td_error in zip(indices, td_errors):
            self.priorities[index] = abs(td_error) + self.epsilon
    
    def __len__(self):
        return len(self.buffer)


def select_action(network, state, epsilon, num_actions=6, env=None):
    """Epsilon-greedy action selection with action masking."""
    valid_actions = env.get_valid_actions() if env else list(range(num_actions))

    if random.random() < epsilon:
        return random.choice(valid_actions)

    numeric_state = cp.array(state.reshape(1, -1), dtype=cp.float32)
    q_values = network.forward(numeric_state)

    masked_q = cp.full_like(q_values, -cp.inf)
    for action in valid_actions:
        masked_q[0, action] = q_values[0, action]
    return int(cp.argmax(masked_q).get())


def copy_network_weights(source_network, target_network):
    """Copy weights from source to target network."""
    for source_layer, target_layer in zip(source_network.layers, target_network.layers):
        target_layer.weights = source_layer.weights.copy()
        target_layer.biases = source_layer.biases.copy()


def soft_update(source_network, target_network, tau=0.005):
    """Polyak averaging: target = tau*source + (1-tau)*target."""
    for source_layer, target_layer in zip(source_network.layers, target_network.layers):
        target_layer.weights = tau * source_layer.weights + (1 - tau) * target_layer.weights
        target_layer.biases = tau * source_layer.biases + (1 - tau) * target_layer.biases


def network_health(network):
    """Return max absolute weight and bias across the network for divergence checks."""
    weight_max = max(float(cp.max(cp.abs(layer.weights))) for layer in network.layers)
    bias_max = max(float(cp.max(cp.abs(layer.biases))) for layer in network.layers)
    return weight_max, bias_max


CURRICULUM_PHASES = [
    {
        "name": "Phase 1: solo (move + destroy crates)",
        "config": dict(n_enemies=0, enemies_drop_bombs=False, enemy_bomb_prob=0.0),
        "epsilon_start": 1.0,
        "advance_threshold": 1.0,
        "max_episodes": 5000,
    },
    {
        "name": "Phase 2: 1 bombing enemy",
        "config": dict(n_enemies=1, enemies_drop_bombs=True, enemy_bomb_prob=0.2, enemy_skill=0.3),
        "epsilon_start": 0.8,
        "advance_threshold": 0.0,
        "max_episodes": 5000,
    },
    {
        "name": "Phase 3: 2 enemies (no bombs)",
        "config": dict(n_enemies=2, enemies_drop_bombs=False, enemy_bomb_prob=0.0, enemy_skill=0.5),
        "epsilon_start": 0.7,
        "advance_threshold": -1.0,
        "max_episodes": 5000,
    },
    {
        "name": "Phase 4: 2 bombing enemies",
        "config": dict(n_enemies=2, enemies_drop_bombs=True, enemy_bomb_prob=0.3, enemy_skill=0.7),
        "epsilon_start": 0.7,
        "advance_threshold": -2.0,
        "max_episodes": 5000,
    },
    {
        "name": "Phase 5: 3 bombing enemies (final)",
        "config": dict(n_enemies=3, enemies_drop_bombs=True, enemy_bomb_prob=0.4, enemy_skill=0.9),
        "epsilon_start": 0.6,
        "advance_threshold": float("inf"),  # run until time budget
        "max_episodes": 10_000_000,
    },
]


def make_phase_config(phase, max_steps=400, crate_density=0.3, width=11, height=11):
    """Build a GameConfig for a curriculum phase (reward semantics from defaults)."""
    return GameConfig(
        width=width,
        height=height,
        max_steps=max_steps,
        crate_density=crate_density,
        **phase["config"],
    )


def build_network():
    """Construct the DQN architecture (417 -> 512 -> 256 -> 128 -> 6).

    Input dim is auto-probed from obs_builder.size (egocentric 9x9x5 crop = 405
    spatial + 12 scalars = 417). The hidden layers were shrunk to match the
    smaller egocentric input.
    """
    probe = BombermanEnv(config=make_phase_config(CURRICULUM_PHASES[0]))
    obs_size = probe.obs_builder.size
    probe.close()
    layer_definitions = [
        {"type": "dense", "input_size": obs_size, "num_neurons": 512, "activation": "relu"},
        {"type": "dense", "input_size": 512, "num_neurons": 256, "activation": "relu"},
        {"type": "dense", "input_size": 256, "num_neurons": 128, "activation": "relu"},
        {"type": "dense", "input_size": 128, "num_neurons": 6, "activation": "linear"},
    ]
    return DQNetwork(layer_definitions)


@dataclass
class TrainingConfig:
    """Static hyperparameters for a training run (set once, never mutated)."""
    batch_size: int = 64
    gamma: float = 0.99
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    updates_per_step: int = 2
    replay_buffer_size: int = 100000
    min_replay_size: int = 10000
    tau: float = 0.005                 # soft target update rate
    hard_sync_every: int = 1000        # hard target copy interval (steps)
    epsilon_decay: float = 0.9995      # per-episode decay within a phase
    min_epsilon: float = 0.05
    max_wall_time_hours: float = 8.0   # auto-stop budget
    lr_warmup_steps: int = 500         # gentle LR ramp after each phase change
    health_check_every: int = 2000     # steps between divergence checks
    log_every: int = 10
    board_width: int = 11
    board_height: int = 11
    crate_density: float = 0.3
    starting_phase: int = 0
    best_path: str = "bomberman_model_best.npz"
    last_good_path: str = "bomberman_model_lastgood.npz"

    def phase_config(self, phase):
        return make_phase_config(
            phase, crate_density=self.crate_density,
            width=self.board_width, height=self.board_height,
        )


@dataclass
class TrainingState:
    """Mutable progress/bookkeeping state that evolves over the run.

    Bundling this avoids threading 8+ separate loop-local variables through
    every helper function below.
    """
    epsilon: float
    phase_index: int
    phase_episode: int = 0
    episode: int = 0
    total_steps: int = 0
    warmup_remaining: int = 0
    learning_rate: float = 3e-4
    best_avg: float = -float("inf")
    start_time: float = field(default_factory=time.time)
    episode_rewards: list = field(default_factory=list)
    rolling_rewards: deque = field(default_factory=lambda: deque(maxlen=100))
    rolling_steps: deque = field(default_factory=lambda: deque(maxlen=100))
    rolling_bombs: deque = field(default_factory=lambda: deque(maxlen=100))
    rolling_crates: deque = field(default_factory=lambda: deque(maxlen=100))
    rolling_deaths: deque = field(default_factory=lambda: deque(maxlen=100))

    @property
    def elapsed_hours(self):
        return (time.time() - self.start_time) / 3600.0


@dataclass
class EpisodeResult:
    """Outcome of running a single episode."""
    episode_reward: float
    steps: int
    bombs_placed: int
    crates_hit: int
    agent_alive: bool


def setup_training(config: TrainingConfig):
    """Build network/target/optimizer/replay-buffer, restoring from checkpoint if present."""
    network = build_network()
    target_network = build_network()
    if os.path.exists(config.best_path):
        print(f"[Checkpoint] Loading best model from {config.best_path}...")
        load_model(network, config.best_path)
    copy_network_weights(network, target_network)

    target_estimator = TargetEstimator(gamma=config.gamma)
    optimizer = AdamWOptimizer(
        network, weight_decay=config.weight_decay,
        learning_rate=config.learning_rate, max_grad_norm=1.0,
    )
    replay_buffer = PrioritizedReplayBuffer(capacity=config.replay_buffer_size, alpha=0.6)
    return network, target_network, optimizer, target_estimator, replay_buffer


def time_budget_exceeded(state: TrainingState, config: TrainingConfig) -> bool:
    """Check (and report) whether the wall-clock training budget has run out."""
    if state.elapsed_hours >= config.max_wall_time_hours:
        print(f"\n[TIME BUDGET] {state.elapsed_hours:.2f}h elapsed >= "
              f"{config.max_wall_time_hours}h. Stopping.")
        return True
    return False


def apply_lr_warmup(optimizer, state: TrainingState, config: TrainingConfig):
    """Ramp the optimizer's learning rate back up gradually after a reset/phase change."""
    if state.warmup_remaining <= 0:
        return
    ramp = 1.0 - state.warmup_remaining / max(1, config.lr_warmup_steps)
    optimizer.learning_rate = state.learning_rate * max(0.05, ramp)
    state.warmup_remaining -= 1
    if state.warmup_remaining == 0:
        optimizer.learning_rate = state.learning_rate


def maybe_hard_sync(network, target_network, state: TrainingState, config: TrainingConfig):
    """Periodically hard-copy weights to the target network on top of soft updates."""
    if config.hard_sync_every and state.total_steps % config.hard_sync_every == 0:
        copy_network_weights(network, target_network)
        print(f"  [TARGET SYNC] hard copy at step {state.total_steps} "
              f"(soft tau={config.tau}/step, hard every {config.hard_sync_every} steps)")


def check_divergence_and_recover(network, target_network, optimizer,
                                  state: TrainingState, config: TrainingConfig):
    """Detect exploding weights/biases and roll back to the last good checkpoint."""
    if state.total_steps % config.health_check_every != 0:
        return
    weight_max, bias_max = network_health(network)
    if bias_max > 100 or weight_max > 100 or not np.isfinite(bias_max + weight_max):
        print(f"  [DIVERGENCE] step {state.total_steps}: weight_max={weight_max:.1f}, "
              f"bias_max={bias_max:.1f}. Restoring last-good and halving LR.")
        if os.path.exists(config.last_good_path):
            load_model(network, config.last_good_path)
            copy_network_weights(network, target_network)
        optimizer.reset()
        state.learning_rate *= 0.5
        optimizer.learning_rate = state.learning_rate
        state.warmup_remaining = config.lr_warmup_steps


def maybe_train_on_step(network, target_network, target_estimator, optimizer,
                         replay_buffer, state: TrainingState, config: TrainingConfig):
    """Run `updates_per_step` gradient updates once the replay buffer has enough data."""
    if len(replay_buffer) < config.min_replay_size:
        return
    for _ in range(config.updates_per_step):
        apply_lr_warmup(optimizer, state, config)
        train_step(network, target_network, target_estimator,
                   optimizer, replay_buffer, config.batch_size, config.gamma)
        soft_update(network, target_network, config.tau)
        state.total_steps += 1
        maybe_hard_sync(network, target_network, state, config)
        check_divergence_and_recover(network, target_network, optimizer, state, config)


def run_episode(env, network, target_network, target_estimator, optimizer,
                 replay_buffer, state: TrainingState, config: TrainingConfig) -> EpisodeResult:
    """Play one episode to completion, training on each step along the way."""
    obs, info = env.reset(seed=state.episode)
    episode_reward = 0.0
    steps = 0
    bombs_placed = 0
    crates_hit = 0
    prev_bomb_count = info["bomb_count"]
    done = False

    while not done:
        action = select_action(network, obs, state.epsilon, env=env)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if info["bomb_count"] > prev_bomb_count:
            bombs_placed += 1
        prev_bomb_count = info["bomb_count"]
        crates_hit += info["tick"]["crates_destroyed"].get(0, 0)

        episode_reward += reward
        steps += 1
        replay_buffer.push(obs, action, reward, next_obs, float(terminated))
        obs = next_obs

        maybe_train_on_step(network, target_network, target_estimator,
                             optimizer, replay_buffer, state, config)

    return EpisodeResult(
        episode_reward=episode_reward,
        steps=steps,
        bombs_placed=bombs_placed,
        crates_hit=crates_hit,
        agent_alive=info["agent_alive"],
    )


def update_episode_stats(state: TrainingState, result: EpisodeResult, config: TrainingConfig):
    """Decay epsilon and append this episode's outcome to the rolling-window stats."""
    state.epsilon = max(config.min_epsilon, state.epsilon * config.epsilon_decay)
    state.episode_rewards.append(result.episode_reward)
    state.rolling_rewards.append(result.episode_reward)
    state.rolling_steps.append(result.steps)
    state.rolling_bombs.append(result.bombs_placed)
    state.rolling_crates.append(result.crates_hit)
    state.rolling_deaths.append(0 if result.agent_alive else 1)
    state.phase_episode += 1
    state.episode += 1


def maybe_save_checkpoint(network, state: TrainingState, config: TrainingConfig):
    """Save best/last-good checkpoints when the rolling-average reward improves."""
    window_full = len(state.rolling_rewards) == state.rolling_rewards.maxlen
    avg_reward = float(np.mean(state.rolling_rewards))
    if window_full and avg_reward > state.best_avg:
        state.best_avg = avg_reward
        save_model(network, config.best_path)
        save_model(network, config.last_good_path)


def log_progress(network, state: TrainingState, result: EpisodeResult, phase: dict):
    """Print a one-line training-progress summary."""
    avg_reward = float(np.mean(state.rolling_rewards))
    avg_steps = float(np.mean(state.rolling_steps))
    weight_max, bias_max = network_health(network)
    total_bombs = sum(state.rolling_bombs)
    total_crates = sum(state.rolling_crates)
    efficiency = total_crates / max(1, total_bombs)
    suicide_rate = 100.0 * sum(state.rolling_deaths) / max(1, len(state.rolling_deaths))
    print(f"Ep {state.episode} | phase {state.phase_index+1} | R {result.episode_reward:6.2f} | "
          f"avgR(100) {avg_reward:6.2f} | avgSteps {avg_steps:5.1f} | "
          f"eps {state.epsilon:.3f} | eff {efficiency:.2f} | suicide {suicide_rate:4.0f}% | "
          f"bias_max {bias_max:.2f} | t {state.elapsed_hours:.2f}h")


def maybe_advance_curriculum(env, phase: dict, optimizer, state: TrainingState, config: TrainingConfig):
    """Check pass/cap conditions and, if met, transition to the next curriculum phase.

    Returns (env, phase) — possibly unchanged, possibly a fresh env on a new phase.
    """
    avg_reward = float(np.mean(state.rolling_rewards)) if state.rolling_rewards else -float("inf")
    window_full = len(state.rolling_rewards) == state.rolling_rewards.maxlen
    passed = window_full and avg_reward >= phase["advance_threshold"]
    capped = state.phase_episode >= phase["max_episodes"]

    if (passed or capped) and state.phase_index < len(CURRICULUM_PHASES) - 1:
        reason = "threshold reached" if passed else "episode cap hit"
        state.phase_index += 1
        phase = CURRICULUM_PHASES[state.phase_index]
        print(f"\n=== CURRICULUM: advancing to {phase['name']} ({reason}, avgR={avg_reward:.2f}) ===")

        env.close()
        env = BombermanEnv(config=config.phase_config(phase))
        state.epsilon = phase["epsilon_start"]
        state.phase_episode = 0
        state.warmup_remaining = config.lr_warmup_steps
        optimizer.reset()
        state.rolling_rewards.clear()
        state.rolling_steps.clear()
        state.rolling_bombs.clear()
        state.rolling_crates.clear()
        state.rolling_deaths.clear()
        state.best_avg = -float("inf")  # best is per-phase to avoid stale comparisons

    return env, phase


def finalize_training(network, env, state: TrainingState, config: TrainingConfig):
    """Close the environment and restore the best checkpoint found during training."""
    env.close()
    if os.path.exists(config.best_path):
        load_model(network, config.best_path)
        print(f"[FINAL] Loaded best model (avgR={state.best_avg:.2f}).")


def train_dqn(
    batch_size=64,
    gamma=0.99,
    learning_rate=3e-4,
    weight_decay=0.01,
    updates_per_step=2,
    replay_buffer_size=100000,
    min_replay_size=10000,
    tau=0.005,                 # soft target update rate
    hard_sync_every=1000,      # hard target copy interval (steps)
    epsilon_decay=0.9995,      # per-episode decay within a phase
    min_epsilon=0.05,
    max_wall_time_hours=8.0,   # auto-stop budget
    lr_warmup_steps=500,       # gentle LR ramp after each phase change
    health_check_every=2000,   # steps between divergence checks
    log_every=10,
    board_width=11,
    board_height=11,
    crate_density=0.3,
    starting_phase=0,
):
    """Train DQN with a performance-gated curriculum and overnight safeguards.

    Thin orchestrator: builds config/state, then loops episode -> stats ->
    checkpoint -> log -> curriculum-check until the time budget runs out.
    """
    config = TrainingConfig(
        batch_size=batch_size, gamma=gamma, learning_rate=learning_rate,
        weight_decay=weight_decay, updates_per_step=updates_per_step,
        replay_buffer_size=replay_buffer_size, min_replay_size=min_replay_size,
        tau=tau, hard_sync_every=hard_sync_every, epsilon_decay=epsilon_decay,
        min_epsilon=min_epsilon, max_wall_time_hours=max_wall_time_hours,
        lr_warmup_steps=lr_warmup_steps, health_check_every=health_check_every,
        log_every=log_every, board_width=board_width, board_height=board_height,
        crate_density=crate_density, starting_phase=starting_phase,
    )

    network, target_network, optimizer, target_estimator, replay_buffer = setup_training(config)

    phase = CURRICULUM_PHASES[config.starting_phase]
    state = TrainingState(epsilon=phase["epsilon_start"], phase_index=config.starting_phase)
    env = BombermanEnv(config=config.phase_config(phase))
    print(f"=== CURRICULUM START: {phase['name']} ===")

    while not time_budget_exceeded(state, config):
        result = run_episode(env, network, target_network, target_estimator,
                              optimizer, replay_buffer, state, config)
        update_episode_stats(state, result, config)
        maybe_save_checkpoint(network, state, config)

        if state.episode % config.log_every == 0:
            log_progress(network, state, result, phase)

        env, phase = maybe_advance_curriculum(env, phase, optimizer, state, config)

    finalize_training(network, env, state, config)
    return network, state.episode_rewards


def train_step(network, target_network, target_estimator, optimizer, replay_buffer, batch_size, gamma):
    """Perform one training step with prioritized experience replay."""
    states, actions, rewards, next_states, dones, weights, indices = replay_buffer.sample(batch_size)
    
    current_q_values = network.forward(states)
    current_q_selected = current_q_values[cp.arange(batch_size), actions]
    
    next_q = target_network.forward(next_states)
    targets = target_estimator.compute_targets(rewards, next_q, dones)
    
    targets = cp.clip(targets, -20, 100)
    td_errors = (targets - current_q_selected).get()
    
    loss = network.weighted_loss(current_q_selected, targets, weights)
    
    grad_selected = 2 * weights * (current_q_selected - targets) / batch_size
    grad_output = cp.zeros_like(current_q_values)
    grad_output[cp.arange(batch_size), actions] = grad_selected
    
    network.backward(grad_output)
    optimizer.step()
    
    replay_buffer.update_priorities(indices, td_errors)
    
    return float(loss.get())


def save_model(network, filepath="bomberman_model.npz"):
    """Save network weights and biases to file."""
    data = {}
    for i, layer in enumerate(network.layers):
        data[f'layer_{i}_weights'] = layer.weights.get()
        data[f'layer_{i}_biases'] = layer.biases.get()

    with _MODEL_SAVE_LOCK:
        for attempt, delay in enumerate((0.0, *_MODEL_SAVE_RETRY_DELAYS), start=1):
            if delay:
                time.sleep(delay)
            try:
                np.savez(filepath, **data)
                break
            except OSError:
                if attempt > len(_MODEL_SAVE_RETRY_DELAYS):
                    raise
    print(f"Model saved to {filepath}")


def load_model(network, filepath="bomberman_model.npz"):
    """Load network weights and biases from file."""
    with np.load(filepath) as data:
        for i, layer in enumerate(network.layers):
            layer.weights = cp.array(data[f'layer_{i}_weights'])
            layer.biases = cp.array(data[f'layer_{i}_biases'])
    print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train DQN agent on Bomberman")
    parser.add_argument("--dry-episodes", type=int, default=200,
                        help="Episodes for the dry run")
    parser.add_argument("--hours", type=float, default=8.0,
                        help="Wall-clock training budget in hours")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--width", type=int, default=11, help="Board width")
    parser.add_argument("--height", type=int, default=11, help="Board height")
    parser.add_argument("--crate-density", type=float, default=0.3,
                        help="Fraction of interior cells that are crates (0.0-1.0)")
    parser.add_argument("--starting-phase", type=int, default=0,
                        help="Curriculum phase to start from (0-4)")
    args = parser.parse_args()

    network, rewards = train_dqn(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        max_wall_time_hours=args.hours,
        board_width=args.width,
        board_height=args.height,
        crate_density=args.crate_density,
        starting_phase=args.starting_phase
    )
    print("Training complete!")
    if rewards:
        print(f"Final 10 episode rewards: {rewards[-10:]}")
    save_model(network, "bomberman_model.npz")
