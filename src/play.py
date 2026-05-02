import argparse
import time
import collections

import gymnasium as gym
import numpy as np
import ale_py
import torch

import src.wrappers as wrappers
from src.dqn import DQN

DEFAULT_ENV_NAME = "ALE/Pong-v5"
FPS = 25


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model file to load.")
    parser.add_argument("-e", "--env", default=DEFAULT_ENV_NAME, help=f"Environment name to use, default={DEFAULT_ENV_NAME}.")
    parser.add_argument("-r", "--record", help="Directory for video (RecordVideo).")
    parser.add_argument("--no-vis", default=True, dest="vis", help="Disable visualization.", action="store_false")
    args = parser.parse_args()

    render_mode = None
    if args.record:
        render_mode = "rgb_array"
    elif args.vis:
        render_mode = "human"

    env = wrappers.make_env(args.env, render_mode=render_mode)

    if args.record:
        env = gym.wrappers.RecordVideo(env, video_folder=args.record, episode_trigger=lambda ep: True)

    obj = torch.load(args.model, map_location="cpu", weights_only=False)

    net = DQN(env.observation_space.shape, env.action_space.n)

    if isinstance(obj, dict) and "net" in obj:
        net.load_state_dict(obj["net"])
    else:
        net.load_state_dict(obj)

    net.eval()

    state, info = env.reset()
    total_reward = 0.0
    c = collections.Counter()

    while True:
        start_ts = time.time()

        if args.vis:
            env.render()

        state_a = np.asarray(state)[None, ...]
        state_v = torch.tensor(state_a, dtype=torch.float32)

        q_vals = net(state_v).detach().cpu().numpy()[0]
        action = int(np.argmax(q_vals))
        c[action] += 1

        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += float(reward)

        if done:
            break

        if args.vis:
            delta = 1 / FPS - (time.time() - start_ts)
            if delta > 0:
                time.sleep(delta)

    print(f"Total reward: {total_reward:.2f}")
    print("Action counts:", c)
    env.close()
