import torch
import numpy as np

from src.experience_buffer import Experience


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        obs, info = self.env.reset()
        self.state = obs
        self.total_reward = 0.0

    @torch.no_grad()
    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None

        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_a = np.asarray(self.state)[None, ...]
            state_v = torch.tensor(state_a, dtype=torch.float32, device=device)
            q_vals_v = net(state_v)
            action = int(torch.argmax(q_vals_v, dim=1).item())

        new_state, reward, terminated, truncated, info = self.env.step(action)
        is_done = terminated or truncated

        self.total_reward += float(reward)

        exp = Experience(self.state, action, float(reward), is_done, new_state)
        self.exp_buffer.append(exp)

        self.state = new_state

        if is_done:
            done_reward = self.total_reward
            self._reset()

        return done_reward
