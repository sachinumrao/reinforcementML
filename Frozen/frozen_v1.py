#import dependencies
from __future__ import print_function
import gym
import numpy as np
import time

#load the environment
env = gym.make('FrozenLake-v0')
s = env.reset()
print("Initial State: \n",s)
env.render()

print(env.action_space.n)
print(env.observation_space.n)

#epsilon-greedy approach
def epsilon_greedy(Q,s,na):
    epsilon = 0.1
    p = np.random.uniform(low=0,high=1)
    if(p>epsilon):
        return np.argmax(Q[s,:])
    else:
        return env.action_space.sample()

#Q-learning implementation
nrows = env.observation_space.n
ncols = env.action_space.n
Q = np.zeros([nrows,ncols])

#set hyper-parameters
lr = 0.5
y = 0.9
eps = 100000

for i in range(eps):
    s = env.reset()
    t = False

    #code for a single episode
    while(True):
        a = epsilon_greedy(Q,s,env.action_space.n)

        #recieve new_state, reward, done and info from environment
        s_,r,t,_ = env.step(a)
        
        #take further step by checking reward
        #if reward is zeros and game is done, put highly negative reward
        if (r==0):
            if(t==True):
                r = -5
                Q[s_] = np.ones(ncols)*r
            else:
                #if game is not done, reward is -1 to avoid longer path
                r = -1
        
        #if reward is 1 i.e. we reached goal, set high reward
        if(r==1):
            r = 100
            Q[s_] = np.ones(ncols)*r

        #Bellman Update Step
        Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s_,a])- Q[s,a])
        
        #update the state
        s = s_

        #if episode is done, finish the episode
        if(t==True):
            break

print("Q-Table")
print(Q)
print("\n")

#check learning of agent
s = env.reset()
env.render()
while(True):
    a = np.argmax(Q[s])
    s_,r,t,_ = env.step(a)
    print("############################")
    env.render()
    s = s_
    if(t==True):
        break

    

