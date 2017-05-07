"""
Translated from https://webdocs.cs.ualberta.ca/~sutton/MountainCar/MountainCar1.cp
Algorithm described at https://webdocs.cs.ualberta.ca/~sutton/book/ebook/node89.html
Some minor adjustments to constants were made to make the program work on environments
besides Mountain Car.
"""

import random
import math
import numpy as np
import gym
import sys

np.random.seed(0)

env = gym.make(sys.argv[1])
outdir = sys.argv[2]

initial_epsilon = 0.1 # probability of choosing a random action (changed from original value of 0.0)
alpha = 0.5 # learning rate
lambda_ = 0.9 # trace decay rate
gamma = 1.0 # discount rate
N = 30000 # memory for storing parameters (changed from original value of 3000)

M = env.action_space.n
NUM_TILINGS = 10
NUM_TILES = 8

def main():

    epsilon = initial_epsilon
    theta = np.zeros(N) # parameters (memory)

    for episode_num in range(2000):
        print( episode_num, episode(epsilon, theta, env.spec.timestep_limit))
        epsilon = epsilon * 0.999 # added epsilon decay

    env.monitor.close()

def episode(epsilon, theta, max_steps):
    Q = np.zeros(M) # action values
    e = np.zeros(N) # eligibility traces
    F = np.zeros((M, NUM_TILINGS), dtype=np.int32) # features for each action

    def load_F(observation):
        state_vars = []
        for i, var in enumerate(observation):
            range_ = (env.observation_space.high[i] - env.observation_space.low[i])
            # in CartPole, there is no range on the velocities, so default to 1
            if range_ == float('inf'):
                range_ = 1
            state_vars.append(var / range_ * NUM_TILES)

        for a in range(M):
            F[a] = get_tiles(NUM_TILINGS, state_vars, N, a)

    def load_Q():
        for a in range(M):
            Q[a] = 0
            for j in range(NUM_TILINGS):
                Q[a] += theta[F[a,j]]

    observation = env.reset()
    load_F(observation)
    load_Q()
    action = np.argmax(Q) # numpy argmax chooses first in a tie, not random like original implementation
    if np.random.random() < epsilon:
        action = env.action_space.sample()

    step = 0
    while True:
        step += 1

        e *= gamma * lambda_
        for a in range(M):
            v = 0.0
            if a == action:
                v = 1.0

            for j in range(NUM_TILINGS):
                e[F[a,j]] = v

        observation, reward, done, info = env.step(action)
        delta = reward - Q[action]
        load_F(observation)
        load_Q()
        next_action = np.argmax(Q)
        if np.random.random() < epsilon:
            next_action = env.action_space.sample()
        if not done:
            delta += gamma * Q[next_action]
        theta += alpha / NUM_TILINGS * delta * e
        load_Q()
        if done or step > max_steps:
            break
        action = next_action
    return step


main()
