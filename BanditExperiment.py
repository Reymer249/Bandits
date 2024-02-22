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

from tqdm import tqdm  # DELETE later


def run_repetitions(
        num_actions: int,
        num_timesteps: int,
        n_rep: int,
        policy_type,
        **kwargs
):
    total_reward = np.zeros(num_timesteps)
    for _ in tqdm(range(n_rep)):
        average_reward = np.zeros(num_timesteps)
        cum_reward = 0
        bandit = BanditEnvironment(n_actions=num_actions)
        if policy_type == OIPolicy:
            policy = policy_type(
                n_actions=num_actions,
                initial_value=kwargs["initial_value"],
                learning_rate=kwargs["learning_rate"]
            )
        else:
            policy = policy_type(n_actions=num_actions)
        for timestep in range(1, num_timesteps):
            if policy_type == EgreedyPolicy:
                action_number = policy.select_action(epsilon=kwargs["epsilon"])
            elif policy_type == UCBPolicy:
                action_number = policy.select_action(c=kwargs["c"], t=timestep)
            else:
                action_number = policy.select_action()
            reward = bandit.act(a=action_number)
            cum_reward += reward
            policy.update(action_number=action_number, reward=reward)
            average_reward[timestep] = cum_reward / (timestep + 1)
        total_reward += average_reward

    return total_reward / n_rep


def experiment(n_actions, n_timesteps, n_repetitions, smoothing_window):
    # To Do: Write all your experiment code here

    # Assignment 1: e-greedy
    egreedy_graph = LearningCurvePlot(title="e-greedy learning curves")
    epsilons = [0.01, 0.05, 0.1, 0.25]
    for epsilon in epsilons:
        curve = run_repetitions(
            num_actions=n_actions,
            num_timesteps=n_timesteps,
            n_rep=n_repetitions,
            policy_type=EgreedyPolicy,
            epsilon=epsilon
        )
        egreedy_graph.add_curve(y=smooth(y=curve, window=smoothing_window), label=f"e={epsilon}")
    egreedy_graph.save(name="egreedy_epsilons_plot.png")

    # Assignment 2: Optimistic init
    optimistic_graph = LearningCurvePlot(title="optimistic learning curves")
    initial_values = [0.1, 0.5, 1.0, 2.0]
    for initial_value in initial_values:
        curve = run_repetitions(
            num_actions=n_actions,
            num_timesteps=n_timesteps,
            n_rep=n_repetitions,
            policy_type=OIPolicy,
            initial_value=initial_value,
            learning_rate=0.1
        )
        optimistic_graph.add_curve(
            y=smooth(y=curve, window=smoothing_window),
            label=f"init_v={initial_value}"
        )
    optimistic_graph.save(name="optimistic_plot.png")

    # Assignment 3: UCB
    ucb_graph = LearningCurvePlot(title="UCB learning curves")
    c_values = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
    for c_value in c_values:
        curve = run_repetitions(
            num_actions=n_actions,
            num_timesteps=n_timesteps,
            n_rep=n_repetitions,
            policy_type=UCBPolicy,
            c=c_value
        )
        ucb_graph.add_curve(
            y=smooth(y=curve, window=smoothing_window),
            label=f"c={c_value}"
        )
    ucb_graph.save(name="ucb_plot.png")

    # Assignment 4: Comparison

    pass


if __name__ == '__main__':
    # experiment settings
    n_actions = 10
    n_repetitions = 500
    n_timesteps = 1000
    smoothing_window = 31

    experiment(n_actions=n_actions, n_timesteps=n_timesteps,
               n_repetitions=n_repetitions, smoothing_window=smoothing_window)
