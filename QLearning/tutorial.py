import itertools
import random
from collections import deque
from typing import List

import gym
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm


class StateMapper:
    """
    Class provides functionality to convert continuous observation space into
    discrete state and get index of discrete state
    """

    def __init__(self, low, high, n_bins=50):
        self.low = low
        self.high = high
        self.n_bins = n_bins
        self.step_size = (self.high - self.low) / self.n_bins

    def get_bin_id(self, val):
        return (val - self.low) // self.step_size


class EnvMapper:
    """
    Class provides functionality to combine ids for each observation dim
    into a single Q-table index
    """

    def __init__(self, bin_list: List[int]):
        self.bin_list = bin_list
        self.ids2state_map = self._build_id2state()

    def __len__(self):
        return len(self.ids2state_map)

    def _build_id2state(self):
        bin_map = []
        for bin_size in self.bin_list:
            state_range = list(range(bin_size))
            bin_map.append(state_range)

        combinations = itertools.product(*bin_map)

        state_map = {}
        for idx, state_comb in enumerate(combinations):
            state_map[state_comb] = idx

        return state_map

    def get_state(self, ids: List[int]) -> int:
        ids = tuple(ids)
        return self.ids2state_map[ids]


class SimpleReplayBuffer:
    """
    Simple buffer class to store the agents experience in environment
    """

    def __init__(self, size: int):
        self.size = size
        self.data = deque([], maxlen=self.size)

    def push(self, list_experience):
        for exp in list_experience:
            self.data.append(exp)

    def sample(self, sample_size):
        if sample_size > len(self.data):
            sample_size = len(self.data)
        return random.sample(self.data, sample_size)


class Agent:
    """
    Implements the agent which accepts an environments and learn to act in it.
    """

    def __init__(self, env_name):
        self.env_name = env_name
        self.env = gym.make(self.env_name)
        _ = self.env.reset()

        self.buffer = SimpleReplayBuffer(10000)

        # get state mappers
        states_high = self.env.observation_space.high
        states_low = self.env.observation_space.low
        n_bins = [30] * len(states_high)
        self.mappers = []

        for high, low, bins in zip(states_high, states_low, n_bins):
            mapper = StateMapper(low, high, n_bins=bins)
            self.mappers.append(mapper)

        self.env_mapper = EnvMapper(n_bins)

        self.n_actions = self.env.action_space.n
        self.n_states = len(self.env_mapper)
        self.Q = np.random.uniform(
            low=-1.0, high=0.0, size=(self.n_states, self.n_actions)
        )
        self.eps = 0.20
        self.gamma = 0.95
        self.alpha = 0.9
        self.n_episodes = 10000
        self.reduction = self.eps / self.n_episodes

    def play(self, is_training=True, is_random=False):
        """
        This function plays the one episode
        """

        # play the episode
        obs = self.env.reset()
        done = False
        experience = []
        cum_reward = 0.0
        steps = 0
        while not done:
            state_bins = [
                self.mappers[idx].get_bin_id(obs_)
                for idx, obs_ in enumerate(obs)
            ]
            state_id = self.env_mapper.get_state(state_bins)
            action = self.get_action(
                state_id, is_training, is_random=is_random
            )
            new_obs, reward, done, info = self.env.step(action)
            if reward != -1.0:
                print(reward)
            experience.append((obs, action, reward, new_obs, done, info))
            cum_reward += reward
            steps += 1
            obs = new_obs

        # save the episode experience
        self.buffer.push(experience)

        # return episode stats
        return {"steps": steps, "total_reward": cum_reward}

    def get_action(
        self, state_id: int, is_training: bool, is_random=False
    ) -> int:
        """
        This function select an action to take based on epsilon greedy policy
        """

        if is_random:
            action = self.env.action_space.sample()
            return action

        prob = np.random.random()

        if prob < self.eps and is_training:
            # select random action
            action = self.env.action_space.sample()

        else:
            # select action from q-table
            action = np.argmax(self.Q[state_id, :])

        return action

    def learn(self):
        """
        This function implements the bellman equation for updating the q-table
        from the experiences stored in replay buffer
        """

        # get sample from experience
        sample_size = 500
        samples = self.buffer.sample(sample_size)

        for exp_tup in samples:
            obs = exp_tup[0]
            new_obs = exp_tup[3]
            reward = exp_tup[2]
            action = exp_tup[1]

            state_bins = [
                self.mappers[idx].get_bin_id(obs_)
                for idx, obs_ in enumerate(obs)
            ]
            state_id = self.env_mapper.get_state(state_bins)

            new_state_bins = [
                self.mappers[idx].get_bin_id(obs_)
                for idx, obs_ in enumerate(new_obs)
            ]
            new_state_id = self.env_mapper.get_state(new_state_bins)

            # bellman optimality update on q-table
            self.Q[state_id, action] = (1 - self.alpha) * self.Q[
                state_id, action
            ] + self.alpha * (
                reward + self.gamma * np.max(self.Q[new_state_id, :])
            )

    def test_policy(self, is_random=False):
        """
        This function make agent play n_episodes using trained policy and gets
        steps and rewards
        """

        print("Testing the agent...")
        test_episodes = 10
        cum_steps = 0
        cum_reward = 0.0
        for i in tqdm(range(test_episodes)):
            run_stats = self.play(is_training=False)
            cum_steps += run_stats["steps"]
            cum_reward += run_stats["total_reward"]

        print("Avg. Steps: ", cum_steps / test_episodes)
        print("Avg. Reward: ", cum_reward / test_episodes)

    def train(self):
        """
        This function trains the agent by iteratively playing and updating
        q-table
        """

        steps = []
        print("Training the agent...")
        for i in tqdm(range(self.n_episodes)):
            # let agent play
            episode_stats = self.play()
            steps.append(episode_stats["steps"])

            if i % 10 == 0:
                self.eps = self.eps - i * self.reduction

            # learn from experience
            self.learn()

        plt.plot(steps)
        plt.show()


def test_env():
    """
    Test the environment by playing with random actions
    """
    env = gym.make("MountainCar-v0")
    _ = env.reset()
    done = False

    while not done:
        action = 2
        obs, reward, done, info = env.step(action)
        print("Obseravtion: ", obs, " Reward: ", reward)
        env.render()

    env.close()


def print_env_info():
    """
    Get the environment spaces information
    """
    env = gym.make("MountainCar-v0")
    _ = env.reset()

    print(env.observation_space.high)
    print(env.observation_space.low)
    print(env.action_space)


def main():
    ENV_NAME = "MountainCar-v0"

    # create agent
    agent = Agent(ENV_NAME)

    # train the agent
    agent.train()

    # evaluate the agent
    agent.test_policy(is_random=True)


if __name__ == "__main__":
    main()

    # test_env()
    # print_env_info()
