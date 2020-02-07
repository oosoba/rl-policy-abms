# Multi-agent RL for ABMs
[[toc]]

RL applied to a selection of policy ABMs

#### Architecting Environments
- Architecture of [OpenAI's gym](./reading-materials/GymArchitecture[openAI-gym].pdf)
  - Useful list of test [gym envs](https://github.com/openai/gym/wiki/Table-of-environments)
- Deepmind [hanabi multi-agent env](https://github.com/deepmind/hanabi-learning-environment)
  - Paper laying out the [hanabi challenge](./reading-materials/DeepMind\ Hanabi\ Multi-agent\ Challenge\ 1902.00506.pdf)

### Requirements list (python)
- tensorflow (==1.15)
- tensorboard (==1.15)
- tensorflow_probability (==0.0.7)
- 3rd Party Open-Source
    - [gym](https://github.com/openai/gym) (OpenAI)
    - python-igraph
      - > conda install -c conda-forge python-igraph

#### Algos
Model | Dev state | Benchmark Envs
|----|:---|----|
RF	| done | Cartpole + MinGame
AC |	done | Cartpole + MinGame
Q-learning	| initial done [AJ] | Taxi
Coordinated RF/RFB |	done | MinGame
Compete RF/RFB |	done | MinGame
Coordinated AC |	done | MinGame + Flu
Compete AC |	done | MinGame + Flu
> switched to actor-critic as MARL default instead of RF/RFB


 .| CartPole | Minority Game | Flu
|----|:---:|:---:|:---:|
RF	| ✅ | ✅ | ✅
Baselined RF	| ✅ | ✅ | ✅
AC | ✅ | ✅ | ✅
**PPO** | ✅ | ❌ | ❌
Q-learning	| Taxi instead | ❌ | ❌

> see caveat in docstring of [PPO implementation](./embodied_arch/embodied_PPO.py)
