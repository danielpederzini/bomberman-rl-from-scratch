"""Watch trained DQN agent play Bomberman."""

import argparse
import cupy as cp
import numpy as np
import time
import os

from bomberman import BombermanEnv
from bomberman.entities import GameConfig
from model.network import DQNetwork


def load_model(network, filepath="bomberman_model_best.npz"):
    """Load network weights and biases from file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}. Run train_agent.py first.")
    data = np.load(filepath)
    for index, layer in enumerate(network.layers):
        layer.weights = cp.array(data[f'layer_{index}_weights'])
        layer.biases = cp.array(data[f'layer_{index}_biases'])
    print(f"Loaded trained model from {filepath}")


def play_trained_agent(network=None, num_episodes=5, render_mode="human", fps=8, model_path="bomberman_model_best.npz",
                       n_enemies=3, enemy_chase_prob=0.85, enemy_bomb_prob=0.4, epsilon=0.0,
                       enemy_skill=0.7, width=11, height=11, crate_density=0.3):
    """Run trained agent in the environment with visualization."""

    eval_config = GameConfig(
        width=width,
        height=height,
        n_enemies=n_enemies,
        enemies_drop_bombs=True,
        enemy_bomb_prob=enemy_bomb_prob,
        enemy_chase_prob=enemy_chase_prob,
        enemy_skill=enemy_skill,
        crate_density=crate_density,
        max_steps=400,
    )

    if network is None:
        temp_env = BombermanEnv(config=eval_config, render_mode=None)
        obs_size = temp_env.obs_builder.size
        temp_env.close()
        
        layer_definitions = [
            {"type": "dense", "input_size": obs_size, "num_neurons": 512, "activation": "relu"},
            {"type": "dense", "input_size": 512, "num_neurons": 256, "activation": "relu"},
            {"type": "dense", "input_size": 256, "num_neurons": 128, "activation": "relu"},
            {"type": "dense", "input_size": 128, "num_neurons": 6, "activation": "linear"},
        ]
        network = DQNetwork(layer_definitions)
        
        try:
            load_model(network, model_path)
            bias_max = float(cp.max(network.layers[0].biases))
            bias_mean = float(cp.mean(network.layers[0].biases))
            weight_mean = float(cp.mean(cp.abs(network.layers[0].weights)))
            print(f"[Model loaded] Layer 0 bias max: {bias_max:.2f}, mean: {bias_mean:.2f}")
            print(f"[Model loaded] Layer 0 weight mean abs: {weight_mean:.6f}")
            if bias_max > 1000:
                print("WARNING: Biases exploded! Delete model and retrain.")
            if weight_mean < 0.001:
                print("WARNING: Weights near zero! Network is not learning.")
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            print("Using untrained network (random actions).")
    
    env = BombermanEnv(config=eval_config, render_mode=render_mode, show_state=True, render_fps=fps, network=network)
    
    for episode in range(num_episodes):
        state, info = env.reset(seed=episode)
        episode_reward = 0
        done = False
        step = 0
        total_crates = 0
        total_kills = 0
        
        print(f"\n=== Episode {episode + 1} ===")
        
        while not done:
            state_cp = cp.array(state.reshape(1, -1), dtype=cp.float32)
            q_values = network.forward(state_cp)
            
            valid_actions = env.get_valid_actions()
            if epsilon > 0.0 and np.random.random() < epsilon:
                action = np.random.choice(valid_actions)
            else:
                masked_q = cp.full_like(q_values, -cp.inf)
                for action in valid_actions:
                    masked_q[0, action] = q_values[0, action]
                action = int(cp.argmax(masked_q).get())
            
            action_names = ["UP", "DOWN", "LEFT", "RIGHT", "BOMB", "WAIT"]
            q_list = q_values.get().flatten()
            best_action = action_names[action]
            print(f"Step {step}: Q-values: {q_list.round(3)}, Action: {best_action}, Valid: {valid_actions}")
            
            if env._renderer is not None:
                env._renderer.record_qvalues(q_list.tolist(), action)
            
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step += 1
            
            if env._renderer is not None:
                tick_info = info.get("tick", {})
                total_crates += sum(tick_info.get("crates_destroyed", {}).values()) if "crates_destroyed" in tick_info else 0
                total_kills += sum(len(kills) for kills in tick_info.get("kills", {}).values()) if "kills" in tick_info else 0
                env._renderer.update_episode_metrics(episode_reward, step, total_crates, total_kills)

                components = info.get("reward_components", {})
                for component_name, component_value in components.items():
                    if component_value != 0:
                        env._renderer.record_event(step, component_name, f"{component_value:+.2f}", component_value)
            
            if render_mode == "human":
                time.sleep(1 / fps)
        
        print(f"Episode {episode + 1} finished: Reward={episode_reward:.2f}, Steps={step}, Agent alive={info['agent_alive']}")
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch trained DQN agent play Bomberman")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to play")
    parser.add_argument("--fps", type=int, default=8, help="Rendering speed (frames per second)")
    parser.add_argument("--no-render", action="store_true", help="Run without visualization")
    parser.add_argument("--model", type=str, default="bomberman_model_best.npz", help="Path to trained model file")
    parser.add_argument("--enemies", type=int, default=3, help="Number of enemies (1-5)")
    parser.add_argument("--chase-prob", type=float, default=0.85, help="Enemy chase probability (0.0-1.0)")
    parser.add_argument("--bomb-prob", type=float, default=0.4, help="Enemy bomb probability (0.0-1.0)")
    parser.add_argument("--epsilon", type=float, default=0.0, help="Exploration epsilon (0.0 = greedy, 1.0 = random)")
    parser.add_argument("--enemy-skill", type=float, default=0.7, help="Enemy skill (0.0 = clumsy, 1.0 = optimal AI)")
    parser.add_argument("--width", type=int, default=11, help="Board width")
    parser.add_argument("--height", type=int, default=11, help="Board height")
    parser.add_argument("--crate-density", type=float, default=0.3,
                        help="Fraction of interior cells that are crates (0.0-1.0)")
    
    args = parser.parse_args()
    
    render_mode = None if args.no_render else "human"
    
    play_trained_agent(
        num_episodes=args.episodes,
        render_mode=render_mode,
        fps=args.fps,
        model_path=args.model,
        n_enemies=args.enemies,
        enemy_chase_prob=args.chase_prob,
        enemy_bomb_prob=args.bomb_prob,
        epsilon=args.epsilon,
        enemy_skill=args.enemy_skill,
        width=args.width,
        height=args.height,
        crate_density=args.crate_density,
    )
