import time
import gym
import numpy as np
from stable_baselines3 import A2C, PPO

def eval_model(model, env, n_games=10):
    
    for ep in range(n_games):
        obs = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            env.render()
            # print(f"Reward: {reward}")
            

def main():
    env = gym.make("LunarLander-v2")
    env.reset()
    
    # print("Sample Action Space: ", env.action_space.sample())
    # print("Sample Observation Space: ", env.observation_space.shape)
    # print("Sample Observation: ", env.observation_space.sample())
    
    # define model
    model = PPO("MlpPolicy", env, verbose=1)
    
    # train model
    model.learn(total_timesteps=2_00_000)
    
    # evaluate model
    print("Evaluating model...")
    time.sleep(5)
    eval_model(model, env)
    
    
if __name__ == "__main__":
    main()
