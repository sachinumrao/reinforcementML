import gym
import time
from DeepQLSTM import Agent
import numpy as np
# import matplotlib.pyplot as plt

print("Starting the simulation...")
env = gym.make("SpaceInvaders-v0")

# Agent params
batch_size = 128
seq_len = 10
img_counts = batch_size + seq_len - 1

agent = Agent(img_counts,
              batch_size,
              seq_len,
              gamma=0.99,
              eps=0.95,
              alpha=0.002,
              max_mem=5000,
              replace=None)

print("Environment is built...")


score_history = []
num_games = 400

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
        if (agent.mem_counter+1) % 20 == 0:
            print("Memory Counter: ", agent.mem_counter+1)

        if agent.mem_counter < seq_len:
            action = np.random.choice(agent.action_space)
        else:
            action = agent.choose_action(frames) 
            frames.pop(0)

        # For training for black and white images use below line
        # frames.append(np.mean(obs[15:200, 30:125], axis=2))
        frames.append(obs[15:200, 30:125])

        obs_, reward, done, info = env.step(action)
        score += reward

        # Extra penalty for dying
        if done and info['ale.lives'] == 0:
            reward = -100

        # Store data into agent's memory
        agent.store_transition(obs[15:200, 30:125], action, reward,
                               obs_[15:200, 30:125])

        obs = obs_

    t2 = time.time()
    print("Data Collection Process Completed...")
    print("Time Taken in Data Collection: ", t2-t1)

    print("Game: ", i+1, " Game Length: ", agent.mem_counter)

    # Make agent learn from stored data
    t3 = time.time()
    agent.learn()
    t4 = time.time()
    print("Time Taken in Learning: ", t4-t3)
    print("Score: ", score)

    # Save score
    score_history.append(score)

# plt.plot(score_history)
# plt.show()

# Test the model by simulating game play
test_frames = []
test_games = 10
test_score_hist = []
for i in range(test_games):
    print("Test Game: ", i)
    obs = env.reset()
    done = False
    agent.mem_counter = 0
    test_score = 0

    while not done:
        if agent.mem_counter < seq_len:
            action = np.random.choice(agent.action_space)
        else:
            action = agent.choose_action(test_frames) 
            test_frames.pop(0)

        env.render()
        test_frames.append(obs[15:200, 30:125])
        obs_, reward, done, info = env.step(action)
        obs = obs_
        test_score += reward
        agent.mem_counter += 1
    print("Score: ", score)
    test_score_hist.append(test_score)

print("Avg Test Score: ", sum(test_score_hist)/len(test_score_hist))
