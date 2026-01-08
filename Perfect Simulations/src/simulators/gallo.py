#import itertools
import random
import math
import numpy as np

from src.utils import LazyU
from src.utils.context_tree import ReferenceContextTree

class GalloSimulator:
    """
    Perfect simulation for chains of infinite order using
    the Gallo (2009) regenerative construction with context trees.

        P(X_0 = x | past) = F(x | context(past))

    where context(past) is determined by a reference string and lag function.
    """

    def __init__(
        self,
        alpha,
        epsilon,
        alphabet,
        reference_string,
        max_depth=8,
    ):
        self.alpha = alpha
        self.eps = epsilon
        self.alphabet = alphabet
        self.reference_string = tuple(reference_string)
        self.max_depth = int(max_depth)
        self.transition_func = SubexpARTransitionModel(alphabet=self.alphabet, alpha=self.alpha).transition_prob

        self.context_tree = ReferenceContextTree(
            alphabet=self.alphabet,
            reference_string=self.reference_string,
            lag_function=self.lag_function,
            max_depth=self.max_depth,
        )


    def lag_function(self,k):
        C_eps = ((self.eps)**(len(self.reference_string)))/len(self.reference_string)
        return math.exp(-k**C_eps)

    def update_func(self, u, past):
        context = self.context_tree.find_context(past)

        upper_left = self.eps * len(self.alphabet)
        # Use context-dependent transitions if context found
        if context is not None and (context in self.context_tree.contexts):
            for symbol in self.alphabet:
                prob = self.transition_func(symbol, context)
                if upper_left <= u < upper_left + prob:
                    return symbol
                upper_left += prob
            return self.alphabet[-1]  # fallback
       
        if context is None:
   
            for i, symbol in enumerate(self.alphabet):
                if i*self.eps <= u < (i + 1)*self.eps:
                    next_alphabet = symbol
                    return next_alphabet
                else:
                    next_alphabet = None
            
            if next_alphabet is None:
                return None

    def perfect_sample(self, window):
        m,n = window
        U = LazyU()
        X = {}
        B = set(range(m, n + 1))
        i = m
        eta = m

        while i <= n and self.update_func(U[i], []) in self.alphabet and B:
            X[i] = self.update_func(U[i], [])
            B.discard(i)
            i += 1
        if not B:
            return eta, [X[j] for j in range(m, n + 1)], X[n]
        
        while B:
            print(f"Current i: {i}, B: {B}")
            i -= 1
            if i not in U:
                U[i] = random.uniform(0, 1)
            while self.eps*len(self.alphabet) <= U[i] < 1.0:
                i -= 1
                U[i] = random.uniform(0, 1)
                print(f"Decreased i to {i}, U[i]: {U[i]}")
            X[i] = self.update_func(U[i], [])
            B.add(i)
            t = min(B)
            while t <= n and self.update_func(U[t], [X[j] for j in range(i, t)]) in self.alphabet and B:
                X[t] = self.update_func(U[t], [X[j] for j in range(i, t)])
                B.discard(t)
                t = min(B) if B else t
        eta = i
        return eta, [X[j] for j in range(eta, n + 1)]


class SubexpARTransitionModel:
    def __init__(self, alphabet, alpha=1.5):
        """
        alphabet     : list of possible symbols
        alpha        : decay exponent (>1 for polynomial decay)
        """
        self.alphabet = alphabet
        self.alpha = alpha

    # ---- subexponential (polynomial) decay ----
    def decay_fn(self, i):
        return 1.0 / ((i + 1) ** self.alpha)

    # ---- simple compatibility / feature function ----
    def feature(self, past_symbol, symbol):
        return 1.0 if past_symbol == symbol else 0.0

    # ---- autoregressive score for a symbol ----
    def transition_score(self, symbol, context):
        score = 0.0
        for i, past_symbol in enumerate(context):
            score += self.decay_fn(i) * self.feature(past_symbol, symbol)
        return score

    # ---- softmax probability ----
    def transition_prob(self, symbol, context):
        scores = np.array([
            self.transition_score(s, context)
            for s in self.alphabet
        ])
        scores -= scores.max()  # numerical stability
        probs = np.exp(scores)
        probs /= probs.sum()
        return probs[self.alphabet.index(symbol)]

    # ---- full distribution (optional) ----
    def transition_distribution(self, context):
        scores = np.array([
            self.transition_score(s, context)
            for s in self.alphabet
        ])
        scores -= scores.max()
        probs = np.exp(scores)
        probs /= probs.sum()
        return dict(zip(self.alphabet, probs))
