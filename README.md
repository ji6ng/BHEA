# BHEA: Budgeted Hierarchical Adversarial Attacks for Robust MARL 
This repository implements a budget-aware adversarial attack and robust training framework for multi-agent reinforcement learning (MARL) on the SMAC benchmark (StarCraft Multi-Agent Challenge), built on top of PyMARL.

## Installation instructions

### 1. Install StarCraft II and SMAC

> **Important**  
> SMAC performance is sensitive to the StarCraft II version.  
> Official SMAC experiments use **SC2.4.6**, while this repository uses **SC2.4.10** by default.

```bash
bash install_sc2.sh
```
This will download SC2 into the 3rdparty folder and copy the maps necessary to run over.
### 2. Install Python Dependencies
```shell
pip install -r requirements.txt
```
The requirements.txt file can be used to install the necessary packages into a virtual environment (not recomended).

## Run an experiment 
### Train Budget-Aware Attacker

```shell
bash run_attack.sh
```
This will load a pretrained clean victim, train a budget-aware attacker and save attacker checkpoints to:
```
results/attack_logs/
```
### Robust Victim Training (Adversarial Training)
```shell
bash run_robust.sh
```
This performs iterative co-training train attacker against the current victim, train victim against the accumulated attacker population and repeat for multiple loops.

Outputs are saved to:
```shell
results/robust_logs/
```
### Evaluation
```shell
bash eval.sh
```
Evaluates:

>A fixed victim policy

>A fixed attacker checkpoint

>Under a specified attack budget B and K

Logs are written to:
```shell
results/eval_logs/
```