import gym
import time
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

batch_size = 128
score_history = []
num_games = 100

for i in range(num_games):
    score = 0
    agent.mem_counter = 0
    agent.memory = []
    frames = []

    obs = env.reset()
    done = False

    # Play with current policy and store data
    print("\nGame: ", i+1, " Data Collection Process Begins...")
    t1 = time.time()
    while not done:
        if agent.mem_counter % 100 == 0:
            print("Memory Counter: ", agent.mem_counter)

        if agent.mem_counter < 3:
            action = np.random.choice(agent.action_space)
        else:
            action = agent.choose_action(frames) 
            frames.pop(0)

        frames.append(np.mean(obs[15:200, 30:125], axis=2))
        obs_, reward, done, info = env.step(action)
        score += reward

        # Extra penalty for dying
        if done and info['ale.lives'] == 0:
                reward = -100

        # Store data into agent's memory
        agent.store_transition(np.mean(obs[15:200, 30:125], axis=2),
                                    action,
                                    reward,
                                    np.mean(obs_[15:200, 30:125], axis=2))

        obs = obs_

    t2 = time.time()
    print("Data Collection Process Completed...")
    print("Time Taken in Data Collection: ", t2-t1)
    
    print("Game: ", i+1, " Game Length: ", agent.mem_counter)
    
    # Make agent learn from stored data
    t3 = time.time()
    agent.learn(batch_size)
    t4 = time.time()
    print("Time Taken in Learning: ", t4-t3)
    print("Score: ", score)

    # Save score
    score_history.append(score)

plt.plot(score_history)
plt.show()