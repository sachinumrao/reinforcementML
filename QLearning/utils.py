import random
from collections import namedtuple

import numpy as np

# define structure for buffer
Buffer = namedtuple("Buffer", ["state", "next_state", "action", "reward", "done"])


class ReplayBuffer:
    def __init__(self, buffer_size, seed=99):
        # set seeds
        random.seed(seed)
        np.random.seed(seed)
        self.data = None
        self.curr_size = 0
        self.max_buffer_size = buffer_size

    def add(self):
        pass

    def sample(self):
        pass

    def get_buffer_size(self):
        return self.curr_size
    
    def reset(self):
        self.data = None
        self.curr_size = 0
    
    

    