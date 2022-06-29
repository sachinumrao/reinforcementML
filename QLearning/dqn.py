import random

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

import config
from model import MLPModel
from utils import ReplayBuffer

# set seeds
random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)


class QNetwork:
    def __init__(self, env):
        self.env = env
        self.model = MLPModel()
        self.buffer = ReplayBuffer(config.BUFFER_SIZE, seed=config.SEED)

    def get_action(self):
        pass
    
    def train(self):
        pass


def main():
    env_name = config.ENV_NAME
    env = gym.amke(env_name)

    network = QNetwork(env)
    network.train()

    env.close()


if __name__ == "__main__":
    main()
