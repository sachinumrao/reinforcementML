import numpy as np
import gym
from ddqn_pytc import Agent
import matplotlib.pyplot as plt

if __name__=="__main__":
    env = gym.make('LunarLander-v2')
    num_games = 1000
    load_checkpoints = False

    agent = Agent(gamma=0.99, epsilon=1.0, alpha=5e-4, input_dims=[8], n_actions=4, mem_size=1000,
                  eps_min=0.1, batch_size=64, eps_dec=1e-3, replace=100)

    scores = []
    eps_history = []

    for i in range(num_games):
        done = False
        score = 0
        obs = env.reset()
        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(obs, action, reward, obs_, int(done))
            agent.learn()

        scores.append(score)
        avg_score = np.mean(scores[-100:])
        print("Episode: ",i," Score: %.2f Average Score: %.2f Epsilon: %.2f" %(score,
                                                                               avg_score, agent.epsilon))
        eps_history.append(agent.epsilon)

    plt.plot(scores)
    plt.show()




