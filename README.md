Codebase for **Robust Multi-Agent Reinforcement Learning with Stochastic Adversary**.

These codes are modified version based on repositories from [https://github.com/oxwhirl/pymarl](https://github.com/oxwhirl/pymarl) and [https://github.com/PKU-RL/FOP-DMAC-MACPF/tree/main](https://github.com/PKU-RL/FOP-DMAC-MACPF/tree/main).

## Project Layout

- `src/main.py`: experiment entry point
- `src/run.py`: training and evaluation pipeline
- `src/runners/`: episode rollout logic
- `src/learners/`: learner implementations
- `src/controllers/`: multi-agent controllers
- `src/modules/`: neural network modules and mixers
- `src/adv/`: adversarial attack implementations
- `src/config/`: algorithm and environment configs

## Training on Clean Observations

```shell
python src/main.py --config=qmix --env-config=sc2 with env_args.map_name=8m Number_attack=0
```

## Training with ATSA

```shell
python src/main.py --config=qmix --env-config=sc2 with env_args.map_name=8m Number_attack=8 attack_method=atsa
```

## Evaluation

```shell
python src/main.py --config=qmix --env-config=sc2 with env_args.map_name=8m evaluate=True Number_attack=8 attack_method=atsa checkpoint_path=results/xxx adv_checkpoint_path=results/xxx
```

## Attack Names

- `ATSA`: the stochastic adversary proposed in this paper, *Robust Multi-Agent Reinforcement Learning with Stochastic Adversary*.
- `PAAD`: the multi-agent extension of PA-AD style adversarial attack methods, related to *Who is the Strongest Enemy? Towards Optimal and Efficient Evasion Attacks in Deep RL* and *Robust Training in Multiagent Deep Reinforcement Learning Against Optimal Adversary*.
- `PR`: the robustness regularization baseline related to *Enhancing the Robustness of QMIX Against State-Adversarial Attacks*.

## Notes

- `Number_attack=0` disables adversarial perturbations.
- `checkpoint_path` points to the victim policy checkpoint.
- `adv_checkpoint_path` points to the adversary checkpoint when the selected attack method requires one.

If you find this work interesting, please cite it as follows:

```bibtex
@inproceedings{zhou2025robust,
  title = {Robust Multi-Agent Reinforcement Learning with Stochastic Adversary},
  author = {Ziyuan Zhou and Guanjun Liu and Mengchu Zhou and Weiran Guo},
  booktitle = {Forty-second International Conference on Machine Learning},
  year = {2025},
  url = {https://openreview.net/forum?id=bnhFueOeav}
}
```

Related references:

```bibtex
@article{GUO2024127191,
  title = {Enhancing the Robustness of QMIX Against State-Adversarial Attacks},
  author = {Weiran Guo and Guanjun Liu and Ziyuan Zhou and Ling Wang and Jiacun Wang},
  journal = {Neurocomputing},
  volume = {572},
  pages = {127191},
  year = {2024},
  issn = {0925-2312},
  doi = {https://doi.org/10.1016/j.neucom.2023.127191},
  url = {https://www.sciencedirect.com/science/article/pii/S0925231223013140}
}
```

```bibtex
@article{guo2025robust,
  title = {Robust Training in Multiagent Deep Reinforcement Learning Against Optimal Adversary},
  author = {Weiran Guo and Guanjun Liu and Ziyuan Zhou and Jiacun Wang and Ying Tang and Miaomiao Wang},
  journal = {IEEE Transactions on Systems, Man, and Cybernetics: Systems},
  volume = {55},
  number = {7},
  pages = {4957--4968},
  year = {2025},
  doi = {10.1109/TSMC.2025.3561276}
}
```
