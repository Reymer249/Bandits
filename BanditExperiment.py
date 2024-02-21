#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bandit environment
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
2021
By Thomas Moerland
"""
import numpy as np
from BanditEnvironment import BanditEnvironment
from BanditPolicies import EgreedyPolicy, OIPolicy, UCBPolicy
from Helper import LearningCurvePlot, ComparisonPlot, smooth

from tqdm import tqdm

def run_repetitions(
        n_actions: int,
        n_timesteps: int,
        n_rep: int,
        **kwargs
):
    total_reward = np.zeros(n_timesteps)
    for _ in tqdm(range(n_rep)):
        average_reward = np.zeros(n_timesteps)
        cum_reward = 0
        bandit = BanditEnvironment(n_actions=n_actions)
        policy = EgreedyPolicy(n_actions=n_actions)
        for timestep in range(1, n_timesteps):
            action_number = policy.select_action(epsilon=kwargs["epsilon"])
            reward = bandit.act(a=action_number)
            cum_reward += reward
            policy.update(action_number=action_number, reward=reward)
            average_reward[timestep] = cum_reward / (timestep+1)
        total_reward += average_reward

    return total_reward / n_rep

def experiment(n_actions, n_timesteps, n_repetitions, smoothing_window):
    #To Do: Write all your experiment code here
    
    # Assignment 1: e-greedy
    egreedy_graph = LearningCurvePlot(title="test")
    epsilons = [0.01, 0.05, 0.1, 0.25]
    for epsilon in epsilons:
        curve = run_repetitions(
            n_actions=n_actions,
            n_timesteps=n_timesteps,
            n_rep=n_repetitions,
            smoothing_window=smoothing_window,
            epsilon=epsilon
        )
        egreedy_graph.add_curve(y=curve, label=f"e={epsilon}")
    egreedy_graph.save(name="egreedy_epsilons_plot.png")
    
    # Assignment 2: Optimistic init
    
    # Assignment 3: UCB
    
    # Assignment 4: Comparison
    
    pass

if __name__ == '__main__':
    # experiment settings
    n_actions = 10
    n_repetitions = 500
    n_timesteps = 1000
    smoothing_window = 31
    
    experiment(n_actions=n_actions,n_timesteps=n_timesteps,
               n_repetitions=n_repetitions,smoothing_window=smoothing_window)
