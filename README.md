# Reinforcement Learning – 10-Armed Bandits

This repository contains a project on **k-armed bandits** (k = 10), completed as part of the Reinforcement Learning course at Leiden University (2024).  
We investigate and compare different action-selection policies in a simplified non-sequential MDP setting.

---

## 📖 Project Overview
- **Environment**: 10-armed bandit with Bernoulli-distributed rewards; each arm’s mean reward is sampled from a uniform distribution on [0,1].  
- **Policies studied**:
  - **ε-greedy**: Selects the current best-known arm most of the time, explores randomly with probability ε.  
  - **Optimal Initialization (OI)**: Uses optimistic initial estimates to encourage exploration.  
  - **Upper Confidence Bound (UCB)**: Balances exploration and exploitation using confidence intervals.  
- **Experiments**:
  - Testing different hyperparameters for each policy: ε, initial values, and UCB exploration constant *c*.  
  - Evaluating both **learning curves** (immediate reward) and **performance curves** (average reward over time).  
  - Comparing optimized policies based on average total reward and exploration behavior.

---

## 🔑 Key Findings
- Extreme hyperparameter values perform poorly, either slowing learning or reducing final rewards.  
- Best parameters found:  
  - ε-greedy: ε = 0.05 or 0.1  
  - Optimal Initialization: init value = 0.5  
  - UCB: c = 0.01, 0.05, 0.1, or 0.25  
- **Upper Confidence Bound policy** performs best overall, providing fast exploration and optimal exploitation.  
- ε-greedy performs worst in all considered characteristics; Optimal Initialization is slower initially but improves with tuned parameters.

---

## 📂 Repository Contents
- `Report.pdf` – Full academic report with methodology, results, and discussion.
- `Appendix.pdf` - An appendix for the report.
- `bandit_assignment.pdf` - An assignment statement specifying the tasks to complete.
- `requirements.txt` - Project requirements.
- `plots/` – Visualizations of learning and performance curves.  
- `src/` – Python code for the environment and policy implementations.  

---

## 🚀 Getting Started
Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
