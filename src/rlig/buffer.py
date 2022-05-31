from   rlig.base import Base

from   collections import deque
import random

class ReplayBuffer(Base):

    def __init__(self, maxlen = 5000):
        super().__init__(locals())
        self.buffer = deque(maxlen = maxlen)

    def store(self, *args):
        self.buffer.append(deque)

    def sample(self, n = 1):
        sampler = random.choice if n == 1 else random.choices
        return sampler(self.buffer)