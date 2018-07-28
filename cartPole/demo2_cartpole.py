#import dependencies
import gym

#define parameters
lr = 1e-3
goal_steps = 500
score_requirement = 50
initial_games = 10000

#create gym environment
env = gym.make('CartPole-v0')
env.reset()



#create function to randomly take actions
def random_games():

    for episode in range(5):
        #reset the game
        env.reset()

        for t in range(200):
            #display the environment
            env.render()

            #randomly sample the action
            action = env.action_space.sample()

            #environment takes action
            observation, reward, done, info = env.step(action)

            #check if the game is over
            if done:
                print("One Episode is Over\n")
                break

#start the game
random_games()

