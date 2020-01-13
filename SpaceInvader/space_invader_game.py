import gym
from DeepQModel import DeepQModel, Agent
import numpy as np
import matplotlib.pyplot as plt


print("Starting the simulation...")
env = gym.make("SpaceInvaders-v0")
agent = Agent(gamma=0.99, 
                eps=0.99,
                alpha=0.01,
                max_mem=5000,
                replace=None)

print("Environment is built...")

while agent.mem_counter < agent.mem_size:
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs_, reward, done, info = env.step(action)
        if done and info['ale.lives'] == 0:
            reward = -100

        agent.store_transition(np.mean(obs[15:200, 30:125], axis=2),
            action, reward, np.mean(obs_[15:200, 30:125], axis=2))

        obs = obs_

    print("Memory Initialized...")

    score_history = []
    eps_history = []
    num_games = 10
    batch_size = 32

    for i in range(num_games):
        print('Starting Game: ', i+1, ' Epsilon: %.4f' % agent.eps)
        eps_history.append(agent.eps)
        done = False
        obs = env.reset()
        frames = [np.sum(obs[15:200, 30:125], axis=2)]
        score = 0
        last_action = 0

        while not done:
            if len(frames) == 3:
                action = agent.choose_action(frames)
                frames = []
            else:
                action = last_action

            obs_, reward, done, info = env.step(action)
            score += reward
            frames.append(np.sum(obs[15:200, 30:125], axis=2))

            if done and info['ale.lives'] == 0:
                reward = -100

            agent.store_transition(np.mean(obs[15:200, 30:125], axis=2),
                                    action,
                                    reward,
                                    np.mean(obs_[15:200, 30:125], axis=2)) 
            obs  = obs_
            agent.learn(batch_size)
            last_action = action

        score_history.append(score)
        print("Score: ", score)

    x = [i+1 for i in range(num_games)]
    
    plt.plot(score_history)
    plt.show()

            




