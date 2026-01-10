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
        alphabet,
        reference_string,
        max_depth=8,
    ):
        self.alpha = alpha
        self.alphabet = alphabet
        self.eps = 0.5*np.sqrt((1 - 1/np.exp(0.5)))
        self.C_eps = (-1/len(reference_string)) * np.log(1 - self.eps ** len(reference_string))
        print(f"C_eps: {self.C_eps:.6f}")
        assert self.alpha < self.C_eps, "Alpha must be less than C_eps to ensure convergence."
        print(f"Computed uniform lower bound on transition probabilities: {self.eps:.6f}")
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
        return np.exp(k**self.alpha)
    
    def inverse_lag_function(self, S):
        return (np.log(S))**(1/self.alpha)
    def update_func(self, u, past):
        context = self.context_tree.find_context(past)

        upper_left = self.eps * len(self.alphabet)
        # Use context-dependent transitions if context found
        if context is not None and (context in self.context_tree.contexts):
            for symbol in self.alphabet:
                prob = self.transition_func(symbol, context) - self.eps
                if self.eps + upper_left <= u < upper_left + prob:
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
        m, n = window
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
            return eta, [X[j] for j in range(m, n + 1)]

        while B:
            print(f"Current i: {i}, B: {B}")
            i -= 1

            # <<< ADDED: enforce max reconstruction depth >>>
            if abs(i) > self.max_depth:
                i = -self.max_depth
                B = {j for j in B if j >= i}
                break
            # <<< END ADDED >>>

            while self.eps * len(self.alphabet) <= U[i] < 1.0:
                i -= 1
                B.add(i)
                U.__setitem__(i, random.uniform(0, 1))

            X[i] = self.update_func(U[i], [])
            B.discard(i)

            t = min(B)
            while (
                t <= n
                and self.update_func(
                    U[t],
                    [X[j] for j in sorted(B) if i <= j < t]
                ) in self.alphabet
                and B
            ):
                X[t] = self.update_func(
                    U[t],
                    [X[j] for j in sorted(B) if i <= j < t]
                )
                B.discard(t)
                t = min(B) if B else t

        eta = i
        for j in range(eta, n + 1):
            if j not in X:
                X[j] = self.update_func(U[j], [X[k] for k in range(eta, j)])


        return eta, [X[j] for j in range(eta, n + 1)]


    def analytic_lookback_expectation(self):
        """
        Compute analytic expectation of lookback time.
        """
        a = 1 - self.eps * len(self.alphabet)
        len_w = len(self.reference_string)
        p_w = (self.eps * len(self.alphabet))**len(self.reference_string)

        assert (1 - (1 - p_w)*math.exp(self.alpha)) > 0, "Divergent lookback expectation"
        kappa = 1 / (1 - (1 - p_w)*math.exp(self.alpha))

        return a*(1/p_w + len_w + p_w*kappa)
    
    def user_impatience_bias(self):
        """
        Compute user impatience bias as given in the formula.

        Returns
        -------
        float
            User impatience bias.
        """
        len_w = len(self.reference_string)
        p_w = (self.eps * len(self.alphabet))**len(self.reference_string)
        P_gt_S = (1/p_w + p_w/(1 - (1 - p_w)*math.exp(self.alpha)))/ (self.max_depth - len_w)
        P_gt_S = P_gt_S if P_gt_S < 1 else 1

        bias = (P_gt_S) / (1 - P_gt_S)

        return bias

    # def empirical_lookback_expectation(self):
    #     """
    #     Compute an upper bound on E[L_n] as given in the formula.



    #     Returns
    #     -------
    #     float
    #         Upper bound on E[L_n].
    #     """
    #     len_w = len(self.reference_string)
    #     p_w = (self.eps * len(self.alphabet))**len(self.reference_string)

    #     P_leq_S = 1 - (1 - p_w)**self.max_depth
    #     P_gt_S = (1 - p_w)**self.max_depth

    #     # First term: L_n <= S
    #     term1 = (1 / (p_w - p_w * (1 - p_w)**self.max_depth)) + len_w + min(self.max_depth, p_w/(1 - (1-p_w)*math.exp(self.alpha)))

    #     # Second term: L_n > S
    #     l_inv = self.inverse_lag_function(self.max_depth)
    #     term2_min = min(
    #         -self.max_depth * (1 - p_w)**l_inv / np.log(1 - p_w),
    #         -(np.exp(self.alpha) * (1 - p_w))**l_inv / (self.alpha + np.log(1 - p_w))
    #     )
    #     term2 = len_w / (p_w * (1 - p_w)**self.max_depth) + term2_min

    #     # Expected value bound
    #     E_Ln_bound = P_leq_S * term1 + P_gt_S * term2

    #     return E_Ln_bound


    def empirical_lookback_expectation(self):
        """
        Compute an upper bound on E[L_n] as given in the formula.



        Returns
        -------
        float
            Upper bound on E[L_n].
        """
        len_w = len(self.reference_string)
        p_w = (self.eps * len(self.alphabet))**len(self.reference_string)

        P_leq_S = 1
        P_gt_S = (1/p_w + p_w/(1 - (1 - p_w)*math.exp(self.alpha)))/ (self.max_depth - len_w)
        P_gt_S = P_gt_S if P_gt_S < 1 else 1

        # First term: L_n <= S
        print(f"P_leq_S: {P_leq_S:.6f}, P_gt_S: {P_gt_S:.6f}")
        term1 = 1
        # Second term: L_n > S

        m_n_s = self.max_depth - len_w
        term2_min = conditional_lag_upper_bound(p_w, self.max_depth - len_w, self.lag_function, m_n_s)
        term2 = len_w + (1 / (p_w * P_gt_S)) + term2_min

        # Expected value bound
        E_Ln_bound = P_leq_S * term1 + P_gt_S * term2

        return E_Ln_bound

    


def conditional_lag_upper_bound(p_w, S, lag_fn, max_k=10_000):
    """
    Upper bound on E[ell^w(m_n) | m_n + ell^w(m_n) > S].
    """
    KS = compute_KS(S, lag_fn, max_k)
    base_term = lag_fn(KS + 1)
    increment_term = sup_lag_increment(lag_fn, KS + 1, max_k) / p_w
    return base_term + increment_term


def sup_lag_increment(lag_fn, start_k, max_k=10_000):
    """
    Compute sup_{k >= start_k} (ell^w(k+1) - ell^w(k))
    over a finite window.
    """
    return max(
        lag_fn(k + 1) - lag_fn(k)
        for k in range(start_k, max_k)
    )


def compute_KS(S, lag_fn, max_k=10_000):
    """
    Compute K_S = max{k : k + ell^w(k) <= S}.
    """
    for k in range(1, max_k):
        if k + lag_fn(k) > S:
            return k - 1
    return max_k


    


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
        return np.exp(- (i + 1)**self.alpha)

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
        return max(probs[self.alphabet.index(symbol)], 0.5*np.sqrt((1 - 1/np.exp(0.5))))

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
