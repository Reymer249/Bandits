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


class EgreedyPolicy:

    def __init__(self, n_actions=10):
        self.n_actions = n_actions
        # Initial means and counts
        self.Q_a = np.zeros(n_actions) # non-optimistic
        self.n_a = np.zeros(n_actions)
        # Array for storing the policy (explicitly)
        self.pi_a = np.zeros(n_actions)

    def select_action(self, epsilon):
        greedy_action = np.argmax(self.Q_a)
        for action_number in range(len(self.pi_a)):
            if action_number == greedy_action:
                self.pi_a[action_number] = 1 - epsilon # Exploitation
            else:
                self.pi_a[action_number] = epsilon / (len(self.Q_a) - 1) # Exploration
        return np.random.choice(len(self.pi_a), p=self.pi_a)

    def update(self, action_number, reward):
        # Incremental update rule
        self.n_a[action_number] += 1
        self.Q_a[action_number] = self.Q_a[action_number] + (1 / self.n_a[action_number]) * (reward - self.Q_a[action_number])


class OIPolicy:

    def __init__(self, n_actions=10, initial_value=0.0, learning_rate=0.1):
        self.n_actions = n_actions
        # Initial means
        self.Q_a = np.full(shape=n_actions, fill_value=initial_value)
        # Learning rate for update rule
        self.learning_rate = learning_rate

    def select_action(self):
        # Greedy policy
        return np.argmax(self.Q_a)

    def update(self, action_number, reward):
        # Learning-based update rule
        self.Q_a[action_number] = self.Q_a[action_number] + self.learning_rate*(reward - self.Q_a[action_number])


class UCBPolicy:

    def __init__(self, n_actions=10):
        self.n_actions = n_actions
        # Initial means and counts
        self.Q_a = np.zeros(n_actions) # non-optimistic
        self.n_a = np.zeros(n_actions)
        # Array for storing the policy (explicitly)
        self.pi_a = np.zeros(n_actions)

    def select_action(self, c, t):
        # UCB policy (the implementation treats the estimate as infinity when self.n_a value is 0, regardless runtime error):
        return np.argmax(self.Q_a + c * np.sqrt((np.log(t)) / self.n_a))

    def update(self, action_number, reward):
        self.n_a[action_number] += 1
        # Incremental update rule
        self.Q_a[action_number] = self.Q_a[action_number] + (1 / self.n_a[action_number]) * (reward - self.Q_a[action_number])


def test():
    n_actions = 10
    env = BanditEnvironment(n_actions=n_actions)  # Initialize environment

    pi = EgreedyPolicy(n_actions=n_actions)  # Initialize policy
    a = pi.select_action(epsilon=0.5)  # select action
    r = env.act(a)  # sample reward
    pi.update(a, r)  # update policy
    print("Test e-greedy policy with action {}, received reward {}".format(a, r))

    pi = OIPolicy(n_actions=n_actions, initial_value=1.0)  # Initialize policy
    a = pi.select_action()  # select action
    r = env.act(a)  # sample reward
    pi.update(a, r)  # update policy
    print("Test greedy optimistic initialization policy with action {}, received reward {}".format(a, r))

    pi = UCBPolicy(n_actions=n_actions)  # Initialize policy
    a = pi.select_action(c=1.0, t=1)  # select action
    r = env.act(a)  # sample reward
    pi.update(a, r)  # update policy
    print("Test UCB policy with action {}, received reward {}".format(a, r))


if __name__ == '__main__':
    test()
