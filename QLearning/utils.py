import random
from collections import deque, namedtuple
from typing import Iterable, List, NamedTuple, Tuple

import torch
from torch.utils.data import DataLoader

# define structure for buffer
Experience = namedtuple(
    "Experience", ["state", "action", "reward", "done", "next_state"]
)


class ReplayBuffer:
    """
    Store agent's experiences from the environment
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.data = deque([], maxlen=self.capacity)

    def push(self, *args) -> None:
        """
        Add Experience to queue
        """
        self.data.append(Experience(*args))

    def sample(self, sample_size: int) -> List[Experience]:
        """
        Get a sample of experience of batch_size
        """
        return random.sample(self.data, sample_size)

    def __len__(self) -> int:
        """
        Returns the current size of replay buffer
        """
        return len(self.data)

    def reset(self) -> None:
        """
        Empty the replay buffer
        """
        self.data = deque([], maxlen=self.capacity)


class GymDataLoader(DataLoader):
    """
    Convert replay buffer sample into torch dataloader
    """

    def __init__(self, buffer: ReplayBuffer, sample_size: int = 256) -> None:
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Iterable[NamedTuple]:
        sample = self.buffer.sample(self.sample_size)
        for i in range(len(sample)):
            yield sample[i]
