import numpy as np


rng = np.random.default_rng(42)
class LazyU:
    def __init__(self):
        self.storage = {}   # only store accessed values

    def __getitem__(self, k):
        if k not in self.storage:
            # generate it once
            self.storage[k] = rng.uniform(0, 1)
        return self.storage[k]