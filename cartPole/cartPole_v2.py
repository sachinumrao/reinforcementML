#import libraries
import numpy as np
import gym

#make environment
env = gym.make('CartPole-v0')
env.reset()

#define strategy function
def train_strategy(env):
    num_episodes = 30
    opt_weight = np.array([0, 0, 0, 0])
    max_step = 0
    w_flag = False
    for _ in range(num_episodes):
        obs = env.reset()
        if w_flag == True:
            opt_weight = 0.1*opt_weight + 0.9*curr_weight

        curr_weight = np.random.rand(4)
        gameOn = True
        step = 0
        while gameOn:
            #env.render()
            policy = np.dot(curr_weight,obs)
            if policy >= 0:
                action=1
            else:
                action = 0

            obs, reward, done, info = env.step(action)
            if done:
                print("Steps: ",step)
                if step > max_step:
                    w_flag = True
                    max_step = step
                else:
                    w_flag = False   
                break
            else:
                step = step+1

    return opt_weight

#After training strategy , test it.
def test_policy(env, opt_weight):
    max_trail = 100
    for _ in range(max_trail):
        test_on = True
        test_steps = 0
        obs = env.reset()
        while test_on:
            policy = np.dot(opt_weight,obs)
            if policy >= 0:
                action =1
            else:
                action = 0

            obs, reward, done, info = env.step(action)
            if done:
                print("Test Steps: ",test_steps)
                if(test_steps < 195):
                    return "Failure"
                break
            else:
                test_steps = test_steps+1

    return "Success"

#train the strategy
opt_weight = train_strategy(env)
result = test_policy(env, opt_weight)
print(result)

