from collections import deque

import gym
import numpy as np
import wandb

import config
from agent import A3CAgent

## Enbale wandb loggin
wandb.init(project="bipedalwalker")

##efine utility functions
def get_gym_env(env_name):
    env = gym.make(env_name)
    env.seed(config.SEED)
    observation = env.reset()[0:14]
    input_size = len(observation)
    output_size = len(env.action_space.sample())
    return env, input_size, output_size

def initialize_state(env):
   current_state = deque()
   for i in range(config.FEED_MEM):  
       current_state.append(env.reset()[0:14])
   return current_state

def train(env, agent):
    for episode in range(config.MAX_EPISODES):
        learning_rate = config.INIT_LR #- episode * lr_change/max_episodes
        current_state = initialize_state(env)
        next_state = current_state
        episodic_reward = 0
        done = False
        trajectory = []
        distance = 0
        for t in range(config.T_MAX):
            action, grad = agent.act(np.expand_dims(current_state, axis = 0))
            current_state = next_state
            state, reward, done, info = env.step(action[0])
            distance += state[2]
            episodic_reward += reward
            next_state.pop()
            next_state.append(state[0:14])
            
            
            if done:
                if t>1500 and distance <350:
                    reward += -150
                    episodic_reward += -150
                
                trajectory.append((current_state, action, reward, next_state, grad, done))
                del current_state, next_state, grad
                break 
                
            trajectory.append((current_state, action, reward, next_state, grad, done))
            
        print(f"Episode: {episode}, Step: {t}, Reward: {episodic_reward}, Distance: {distance}")
        wandb.log({"Steps": t,
                    "LR": learning_rate,
                    "Reward": episodic_reward})
        
        
        agent.remember(trajectory)
        del trajectory
        
            
        if episode != 0 and episode%config.TRAIN_FREQ == 0:
            print("Training the Neural Networks ......")
            agent.train_value_model(learning_rate)
            agent.train_policy_model(learning_rate)
            agent.forget()
            agent.save(f"latest")
        

def main():
    # initiate environment
    env_name = config.ENV_NAME
    env, input_size, output_size = get_gym_env(env_name)
    
    # initiate agent
    agent = A3CAgent(config.MEM_SIZE, input_size, output_size, config.GAMMA)
    
    # train the agent
    train(env, agent)
    
    
if __name__ == "__main__":
    main()
