from keras import backend as K
from keras.layers import Activation, Dense, Input
from keras.models import Model
from keras.optimizers import Adam
import numpy as np

class Agen(object):
    def __init__(self, aplha, beta, gamma=0.99, n_actions=4, layer1_size=1024,
                 layer2_size=512, input_dims=8):
        self.gamma = gamma
        self.aplha = aplha
        self.beta = beta
        self.input_dims = input_dims
        self.fc1_dims = layer1_size
        self.fc2_dims = layer2_size
        self.n_actions = n_actions

        self.actor, self.critic, self.policy = self.build_actor_critic_network()
        self.action_space = [i for i in range(self.n_actions)]

    def build_actor_critic_network(self):
        input = Input(shape=(self.input_dims,))
        delta = Input(shape=[1])
        dense1 = Dense()
        dense2 = Dense()
        probs = Dense()
        values = Dense()



