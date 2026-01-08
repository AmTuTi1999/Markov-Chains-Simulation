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
    
    def __setitem__(self, k, v):
        self.storage[k] = v


def transition_prob_lower_bound(alphabet_size, alpha, tol=1e-10, max_terms=10_000):
    """
    Computes a uniform non-zero lower bound on transition probabilities
    for the SubexpARTransitionModel.

    Parameters
    ----------
    alphabet_size : int
        |A|, number of symbols
    alpha : float
        decay exponent (> 0)
    tol : float
        truncation tolerance for the summation
    max_terms : int
        safety cap on number of terms

    Returns
    -------
    float
        uniform lower bound on transition probabilities
    """
    # compute C_alpha = sum_{i=1}^\infty exp(-i^alpha)
    C_alpha = 0.0
    for i in range(1, max_terms + 1):
        term = np.exp(-i ** alpha)
        C_alpha += term
        if term < tol:
            break

    # lower bound: 1 / ((A - 1) + exp(C_alpha))
    return 1.0 / ((alphabet_size - 1) + np.exp(C_alpha))
