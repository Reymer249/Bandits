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


def run_repetitions(
        num_actions: int,
        num_timesteps: int,
        n_rep: int,
        policy_type,
        **kwargs
):
    total_reward = np.zeros(num_timesteps)
    for _ in range(n_rep):
        # Clear bandit environment
        bandit = BanditEnvironment(n_actions=num_actions)
        average_reward = np.zeros(num_timesteps)
        cum_reward = 0
        # Initialise policy, declare correct arguments depending on policy type
        if policy_type == OIPolicy:
            policy = policy_type(
                n_actions=num_actions,
                initial_value=kwargs["initial_value"],
                learning_rate=kwargs["learning_rate"]
            )
        else:
            policy = policy_type(n_actions=num_actions)
        # Run the experiment (num_timesteps-1) times
        # We leave the 0th value as 0 so that it looks more intuitive on a graph
        for timestep in range(1, num_timesteps):
            # Sample next action
            if policy_type == EgreedyPolicy:
                action_number = policy.select_action(epsilon=kwargs["epsilon"])
            elif policy_type == UCBPolicy:
                action_number = policy.select_action(c=kwargs["c"], t=timestep)
            else:
                action_number = policy.select_action()
            # Sample reward
            reward = bandit.act(a=action_number)
            cum_reward += reward
            # Update policy
            policy.update(action_number=action_number, reward=reward)
            # Store average reward
            average_reward[timestep] = cum_reward / timestep
        total_reward += average_reward

    return total_reward / n_rep


def experiment(
        n_actions: int,
        n_timesteps: int,
        n_repetitions: int,
        smoothing_window: int,
        epsilons: list,
        initial_values: list,
        c_values: list
) -> None:
    

    # Assignment 1: e-greedy

    egreedy_graph = LearningCurvePlot(title="E-greedy learning curves based on epsilon hyperparameter")
    epsilons_end_avg_reward = np.zeros(len(epsilons))
    # Run the experiment for each epsilon value
    for i in range(len(epsilons)):
        curve = run_repetitions(
            num_actions=n_actions,
            num_timesteps=n_timesteps,
            n_rep=n_repetitions,
            policy_type=EgreedyPolicy,
            epsilon=epsilons[i]
        )
        egreedy_graph.add_curve(
            y=smooth(y=curve, window=smoothing_window),
            label=f"e={epsilons[i]}"
        )
        epsilons_end_avg_reward[i] = curve[-1]
    egreedy_graph.save(name="egreedy_epsilons_graph.png")


    # Assignment 2: Optimistic init

    optimistic_graph = LearningCurvePlot(title="OI learn. curves based on init. val. hyperparam. (learn. rate = 0.1)")
    initial_values_end_avg_reward = np.zeros(len(initial_values))
    # Run the experiment for each initial value
    for i in range(len(initial_values)):
        curve = run_repetitions(
            num_actions=n_actions,
            num_timesteps=n_timesteps,
            n_rep=n_repetitions,
            policy_type=OIPolicy,
            initial_value=initial_values[i],
            learning_rate=0.1  # fixed for our experiments
        )
        optimistic_graph.add_curve(
            y=smooth(y=curve, window=smoothing_window),
            label=f"init_val={initial_values[i]}"
        )
        initial_values_end_avg_reward[i] = curve[-1]
    optimistic_graph.save(name="optimistic_initials_graph.png")


    # Assignment 3: UCB

    ucb_graph = LearningCurvePlot(title="UCB learning curves based on exploration constant hyperparameter")
    c_values_end_avg_reward = np.zeros(len(c_values))
    # Run the experiment for each value for c
    for i in range(len(c_values)):
        curve = run_repetitions(
            num_actions=n_actions,
            num_timesteps=n_timesteps,
            n_rep=n_repetitions,
            policy_type=UCBPolicy,
            c=c_values[i]
        )
        ucb_graph.add_curve(
            y=smooth(y=curve, window=smoothing_window),
            label=f"c={c_values[i]}"
        )
        c_values_end_avg_reward[i] = curve[-1]
    ucb_graph.save(name="ucb_constants_graph.png")


    # Assignment 4: Comparison

    # Comparison plot

    comparison_plot = ComparisonPlot(title="Average end reward based on alg. and hyperparameter value")
    # Using average rewards found in previous sections
    comparison_plot.add_curve(x=epsilons, y=epsilons_end_avg_reward, label="E-greedy")
    comparison_plot.add_curve(x=initial_values, y=initial_values_end_avg_reward, label="Optimistic Initialization (alpha=0.1)")
    comparison_plot.add_curve(x=c_values, y=c_values_end_avg_reward, label="UCB")
    comparison_plot.save("comparison_graph.png")

    # Optimal cases plot

    optimal_cases_plot = LearningCurvePlot(title="Learning curves of algs. with optimal hyperparameters")
    # Find optimal epsilon for e-greedy
    epsilon_optimal = epsilons[np.argmax(epsilons_end_avg_reward)]
    epsilon_optimal_curve = run_repetitions(
        num_actions=n_actions,
        num_timesteps=n_timesteps,
        n_rep=n_repetitions,
        policy_type=EgreedyPolicy,
        epsilon=epsilon_optimal
    )
    # Add e-greedy to the plot
    optimal_cases_plot.add_curve(
        y=smooth(y=epsilon_optimal_curve, window=smoothing_window),
        label=f"E-greedy (e={epsilon_optimal})"
    )
    # Find optimal initial value for OI
    initial_value_optimal = initial_values[np.argmax(initial_values_end_avg_reward)]
    oi_optimal_curve = run_repetitions(
        num_actions=n_actions,
        num_timesteps=n_timesteps,
        n_rep=n_repetitions,
        policy_type=OIPolicy,
        initial_value=initial_value_optimal,
        learning_rate=0.1  # fixed for our experiments
    )
    # Add OI to the plot
    optimal_cases_plot.add_curve(
        y=smooth(y=oi_optimal_curve, window=smoothing_window),
        label=f"Optimistic init. (init_val={initial_value_optimal})"
    )
    # Find optimal c for UCB
    c_optimal = c_values[np.argmax(c_values_end_avg_reward)]
    ucb_optimal_curve = run_repetitions(
        num_actions=n_actions,
        num_timesteps=n_timesteps,
        n_rep=n_repetitions,
        policy_type=UCBPolicy,
        c=c_optimal
    )
    # Add UCB to the plot
    optimal_cases_plot.add_curve(
        y=smooth(y=ucb_optimal_curve, window=smoothing_window),
        label=f"UCB (c={c_optimal})"
    )
    # Save the plot
    optimal_cases_plot.save(name="optimal_comparison_graph.png")


if __name__ == '__main__':
    # experiment settings
    number_actions = 10
    number_repetitions = 500
    number_timesteps = 1000
    smoothing_w = 31
    epsilons_exp_values = [0.01, 0.05, 0.1, 0.25]
    initial_exp_values = [0.1, 0.5, 1.0, 2.0]
    c_exp_values = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]

    experiment(
        n_actions=number_actions,
        n_timesteps=number_timesteps,
        n_repetitions=number_repetitions,
        smoothing_window=smoothing_w,
        epsilons=epsilons_exp_values,
        initial_values=initial_exp_values,
        c_values=c_exp_values
    )
