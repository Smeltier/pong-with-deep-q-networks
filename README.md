# Pong with Deep Q-Networks (DQN)

Using **Deep Q-Networks (DQN)** to play **Atari Pong** via **Gymnasium + ALE**.

- Repository: `Smeltier/pong-with-deep-q-networks`
- Python used: **Python 3.12.3**
- Default environment: `ALE/Pong-v5`
- Checkpoints directory: `checkpoints/`

## Quick overview

This repo has two main “run” scripts:

- **Training**: `src/train.py`  
  Trains a DQN agent on Pong, periodically saving checkpoints and logging metrics to TensorBoard (via `tensorboardX`).

- **Visualization / Play**: `src/play.py`  
  Loads a checkpoint and runs an episode to **watch the agent play** (human render) or **record a video**.

There is also `src/wrappers.py`, which implements classic Atari preprocessing (frame skipping, max pooling over frames, resizing/grayscale, frame stacking, etc.).

---

## Project structure

- `src/train.py` — DQN training loop (online + target network)
- `src/play.py` — runs one episode using a trained model (render/record)
- `src/dqn.py` — Q-network architecture
- `src/agent.py` — environment interaction (ε-greedy + experience collection)
- `src/experience_buffer.py` — replay buffer
- `src/wrappers.py` — Atari preprocessing wrappers
- `requirements.txt` — pinned dependencies

---

## Dependencies

Dependencies are listed in `requirements.txt`. Key packages include:

- **gymnasium** + **ale-py**: Atari environment (Pong)
- **torch** + **torchvision**: neural network + training
- **opencv-python**: preprocessing (resize) inside wrappers
- **tensorboardX**: training logs (for TensorBoard)
- **numpy**, **tqdm**, etc.

> Note: on some systems, rendering may require extra OS-level dependencies (display/OpenGL). This is OS-dependent.

---

## Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/Smeltier/pong-with-deep-q-networks.git
cd pong-with-deep-q-networks

python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows (PowerShell):
# .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Preprocessing pipeline (wrappers)

`src/wrappers.py` builds the environment with a typical Atari pipeline:

- **MaxAndSkipEnv(skip=4)**: repeats the same action for `skip` frames and returns the *max* of the last 2 frames (reduces flickering).
- **FireResetEnv**: automatically presses “FIRE” actions on reset (required by some Atari games).
- **ProcessFrame84**: converts to grayscale and resizes to **84×84**.
- **ImageToPyTorch**: converts observation layout to **CHW** (channels-first).
- **BufferWrapper(n_steps=4)**: **frame stacking** (4 frames).
- **ScaledFloatFrame**: scales pixels to `[0, 1]`.

Default env used by the scripts: `ALE/Pong-v5`.

---

## Training (`src/train.py`)

The training script includes:

- **Replay Buffer** (`REPLAY_SIZE = 10_000`)
- **Warm-up** before learning (`REPLAY_START_SIZE = 10_000`)
- **Target Network** sync every `SYNC_TARGET_FRAMES = 1000`
- **Linear epsilon decay** from `1.0` to `0.01` across `150_000` frames
- **Loss**: MSE between Q(s,a) and the Bellman target using the target net
- **Checkpoints**:
  - saves “latest” every `SAVE_FREQUENCY = 10` episodes: `checkpoints/Pong-v5-latest.dat`
  - saves “best” when the 100-episode mean reward improves: `checkpoints/Pong-v5-best.dat`

### Important: `--model` argument

`src/train.py` currently **always tries to load** a model via `--model`:

- It calls `torch.load(args.model, ...)` and does not have a fallback if `--model` is missing.
- That means: to start training, you must provide an existing file **or** adjust the script to support training “from scratch”.

### Commands

Train on CPU:

```bash
python -m src.train --model checkpoints/Pong-v5-latest.dat
```

Train with CUDA:

```bash
python -m src.train --cuda --model checkpoints/Pong-v5-latest.dat
```

Specify another environment:

```bash
python -m src.train --env "ALE/Pong-v5" --model checkpoints/Pong-v5-latest.dat
```

### TensorBoard logs

Training uses `tensorboardX.SummaryWriter` and writes logs to the default `runs/` directory (which is in `.gitignore`).

To visualize:

```bash
tensorboard --logdir runs
```

---

## Visualization / Play (`src/play.py`)

`src/play.py`:

- loads a checkpoint (`--model` is required)
- runs a single episode
- by default uses human rendering (if available)
- can **disable visualization** and/or **record video**

### Commands

Run with visualization (default behavior):

```bash
python -m src.play --model checkpoints/Pong-v5-best.dat
```

Disable visualization (useful for headless runs):

```bash
python -m src.play --model checkpoints/Pong-v5-best.dat --no-vis
```

Record video (sets `render_mode="rgb_array"` and wraps with `gym.wrappers.RecordVideo`):

```bash
python -m src.play --model checkpoints/Pong-v5-best.dat --record videos/
```

Select environment:

```bash
python -m src.play --model checkpoints/Pong-v5-best.dat --env "ALE/Pong-v5"
```

At the end it prints:

- `Total reward: ...`
- action counts (a `Counter`) for the chosen actions during the episode

---

## Checkpoints

Both scripts accept two checkpoint formats:

1. **Full training checkpoint** (a dict with `"net"`), also containing:
   - `"tgt_net"`, `"optimizer"`, `"frame_idx"`, `"epsilon"`, `"best_m_reward"`, `"total_rewards"`
2. **Plain PyTorch state dict** (network weights only)

During training, the `.dat` files saved into `checkpoints/` are format (1).

---

## Notes / Possible improvements (optional)

- **Training bootstrap**: `train.py` currently requires `--model`. If you want true “train from scratch”, you can change it so that when `--model` is not provided it skips loading and just starts with randomly initialized weights.
- **REPLAY_START_SIZE == REPLAY_SIZE** (both 10,000): this forces the buffer to fill completely before learning starts. This is valid, but can be slow at the beginning.

---
