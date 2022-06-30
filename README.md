# Hindsight Instruction Prediction from State Sequences (HIPSS)

This repository contains the implementations of our paper [**_Grounding Hindsight Instructions in Multi-Goal Reinforcement Learning for Robotics_**](https://arxiv.org/abs/2204.04308).
Both proposed methods **_HEIR_** and **_HIPSS_** are part of this source code.

```
@article{Roder_GroundingHindsight_2022,
  title = {Grounding {{Hindsight Instructions}} in {{Multi-Goal Reinforcement Learning}} for {{Robotics}}},
  author = {R{\"o}der, Frank and Eppe, Manfred and Wermter, Stefan},
  journal = {arXiv preprint arXiv:2204.04308 [cs]},
  year = {2022},
}
```

## Installation
- `git clone https://github.com/frankroeder/hipss.git`
- pip users: `pip install -r requirements.txt`
- conda users: `conda create --file= conda_env.yaml`

## Training

To reproduce the results of our paper, please have a look at the script `train.sh`

```bash
python train.py n_epochs=20 agent=LCSAC env_name=PandaNLReach2-v0
```

## Enjoy
```bash
python demo.py --demo-path <path to the trial folder>
python demo.py --wandb-url <wandb URI: entity/project/runs/trialid>
```

## Developers

- Copy `example.pyproject.toml` to `pyproject.toml` and adjust the values.
- Install `yapf` for formatting and `pyright` for type-checking etc.
