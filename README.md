# Language-Conditioned Soft Actor-Critic

This repository contains various implementations to solve goal-conditioned and
language-conditioned RL tasks.
Features such as hindsight experience replay for goals and language
instructions, as well as methods for intrinsic motivation are contained.

## Installation
> ⚠️ Access to [LANRO](https://github.com/frankroeder/lanro) is required!

- `git clone https://github.com/frankroeder/lcrl.git`
- pip users: `pip install -r requirements.txt`
- conda users: `conda create --file= conda_env.yaml`

## Training

### Single process
```bash
# goal-conditioned
python train.py n_epochs=10 agent=SAC env_name=PandaReach-v0
# language-conditioned
python train.py n_epochs=20 agent=LCSAC env_name=PandaNLReach2-v0 with_gru=True
```

### Multiprocess
```bash
# goal-conditioned
mpirun -np 4 python -u train.py n_epochs=10 agent=SAC env_name=PandaReach-v0
# language-conditioned
mpirun -np 4 python -u train.py n_epochs=20 agent=LCSAC env_name=PandaNLReach2-v0 with_gru=True
```

## Enjoy
```bash
# here we use normal arguments
python demo.py --demo-path <path to the trial folder>
```

## Developers

- Copy `example.pyproject.toml` to `pyproject.toml` and adjust the values.
- Install `yapf` for formatting and `pyright` for type-checking etc.
