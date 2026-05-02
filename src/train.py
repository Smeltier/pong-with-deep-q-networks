import argparse
import time

import numpy as np
import ale_py
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from src.dqn import DQN
from src.experience_buffer import ExperienceBuffer
from src.agent import Agent
import src.wrappers as wrappers

DEFAULT_ENV_NAME = 'ALE/Pong-v5'
SAVE_FREQUENCY = 10
MEAN_REWARD_BOUND = 19.0
GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10_000
REPLAY_START_SIZE = 10_000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
EPSILON_DECAY_LAST_FRAME = 150_000
EPSILON_START = 1.0
EPSILON_FINAL = 0.01


def calc_loss(batch, net, tgt_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(np.array(states, copy=False), dtype=torch.float32, device=device)
    next_states_v = torch.tensor(np.array(next_states, copy=False), dtype=torch.float32, device=device)

    actions_v = torch.tensor(actions, dtype=torch.long, device=device)
    rewards_v = torch.tensor(rewards, dtype=torch.float32, device=device)
    done_mask = torch.tensor(dones, dtype=torch.bool, device=device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

    with torch.no_grad():
        next_state_values = tgt_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0

    expected_state_action_values = rewards_v + GAMMA * next_state_values
    return nn.MSELoss()(state_action_values, expected_state_action_values)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=False, action='store_true', help='Enable cuda')
    parser.add_argument('--env', default=DEFAULT_ENV_NAME, help=f'Name of the environment, default = {DEFAULT_ENV_NAME}')
    parser.add_argument("-m", "--model", help="Model file to load.")
    args = parser.parse_args()

    device = torch.device('cuda' if args.cuda else "cpu")

    env = wrappers.make_env(args.env)
    net = DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = DQN(env.observation_space.shape, env.action_space.n).to(device)

    writer = SummaryWriter(comment=f'-{args.env}')
    print(net)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_m_reward = None

    def load_checkpoint(path, device):
        try:
            return torch.load(path, map_location=device, weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load {path}: {e}")

    obj = load_checkpoint(args.model, device)

    if isinstance(obj, dict) and "net" in obj:
        checkpoint = obj
        net.load_state_dict(checkpoint["net"])
        tgt_net.load_state_dict(checkpoint.get("tgt_net", checkpoint["net"]))
        optimizer.load_state_dict(checkpoint["optimizer"])
        frame_idx = checkpoint.get("frame_idx", 0)
        best_m_reward = checkpoint.get("best_m_reward", None)
        total_rewards = checkpoint.get("total_rewards", [])
    else:
        net.load_state_dict(obj)
        tgt_net.load_state_dict(net.state_dict())

    while True:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

        reward = agent.play_step(net, epsilon, device=str(device))

        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            m_reward = np.mean(total_rewards[-100:])

            print(f"{frame_idx}: done {len(total_rewards)} games, reward {m_reward}, eps {epsilon}, speed {speed:.2f}")

            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", m_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)

            checkpoint = {
                "net": net.state_dict(),
                "tgt_net": tgt_net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "frame_idx": frame_idx,
                "epsilon": epsilon,
                "best_m_reward": m_reward,
                "total_rewards": total_rewards
            }

            from pathlib import Path
            Path("checkpoints").mkdir(parents=True, exist_ok=True)

            if len(total_rewards) % SAVE_FREQUENCY == 0:
                torch.save(checkpoint, 'checkpoints/Pong-v5-latest.dat')

            if best_m_reward is None or best_m_reward < m_reward:
                if best_m_reward is not None:
                    print(f'Best reward updated {best_m_reward} -> {m_reward}')

                best_m_reward = m_reward
                checkpoint['best_m_reward'] = best_m_reward
                torch.save(checkpoint, 'checkpoints/Pong-v5-best.dat')

            if m_reward > MEAN_REWARD_BOUND:
                print(f'Solved in {frame_idx} frames!')
                break

        if len(buffer) < REPLAY_START_SIZE:
            continue

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device=str(device))
        loss_t.backward()
        optimizer.step()
