import cupy as cp

class TargetEstimator:
    def __init__(self, gamma: float):
        self.gamma = gamma

    def compute_targets(self, rewards: cp.ndarray, next_q_values: cp.ndarray, dones: cp.ndarray) -> cp.ndarray:
        max_next_q_values = cp.max(next_q_values, axis=1)
        targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        return targets
