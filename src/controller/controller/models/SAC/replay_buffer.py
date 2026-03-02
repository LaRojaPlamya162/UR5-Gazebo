import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity, device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        # Pre-allocate memory (QUAN TRỌNG)
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)

    def push(self, s, a, r, s_, d):
        self.state[self.ptr] = s
        self.action[self.ptr] = a
        self.reward[self.ptr] = r
        self.next_state[self.ptr] = s_
        self.done[self.ptr] = d

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)

        # Convert trực tiếp sang torch trên GPU
        return (
            torch.as_tensor(self.state[idx], device=self.device),
            torch.as_tensor(self.action[idx], device=self.device),
            torch.as_tensor(self.reward[idx], device=self.device),
            torch.as_tensor(self.next_state[idx], device=self.device),
            torch.as_tensor(self.done[idx], device=self.device),
        )

    def __len__(self):
        return self.size
