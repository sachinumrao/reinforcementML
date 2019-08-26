import gym
import matplotlib.pyplot as plt
import numpy as np
from policy_grad import Agent

if __name__ == "__main__":
    agent = Agent(alpha=0.0005, input_dims=8, gamma=0.99, n_actions=4,
                layer1_size=64, layer2_size=64)

    env = gym.make('LunarLander-v2')

    score_history = []
    n_episodes = 2000

    for i in range(n_episodes):
        done = False
        score = 0
        obs = env.reset()

        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, info = env.step(action)
            agent.store_transition(obs, action, reward)
            
            obs = obs_
            score += reward

        score_history.append(score)

        agent.learn()

        print("Episode : ",i+1, " Score : ",score, " Mean Score : ", np.mean(score_history[-100:]))

    fileanme = 'lander.png'




