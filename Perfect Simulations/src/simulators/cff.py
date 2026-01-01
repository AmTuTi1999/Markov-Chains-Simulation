import math
from random import random

from src.simulators.perfect_simulator import PerfectSimulator
import numpy as np
from tqdm import tqdm

rng = np.random.default_rng(42)

class BinaryAutoregressiveSimulator(PerfectSimulator):
    """
    Implements the perfect simulation scheme from Section 3
    for the binary autoregression model:

        P(X_0 = 1 | past) = q(theta0 + sum_k theta_k * X_{-k})

    using the regenerative construction of Comets–Fernández–Ferrari (2002).
    """

    def __init__(self, theta0, theta_seq, alphabet=None, max_regen_search_depth=1000):
        self.theta0 = theta0
        self.theta = theta_seq     
        self.G = alphabet if alphabet is not None else [-1, +1]
        self.max_depth = max_regen_search_depth

    # --------------------------------------------------------
    # Model: P(g | history)
    # --------------------------------------------------------

    def q_logistic(self, x):
        return 1.0 / (1.0 + math.exp(-2 * x))

    def conditional_P(self, g, history):
        """
        P(g | w_{-1}, w_{-2}, ...)
        """
        x = self.theta0
        for k, wk in enumerate(history, start=1):
            x += self.theta.get(k, 0.0) * wk
        p = self.q_logistic(x)
        return p if g == +1 else 1 - p

    # --------------------------------------------------------
    # Compute a_k(g|w)  (Section 3 / Section 9.1)
    # --------------------------------------------------------

    def a_k_g_given_w(self, k, g, w_minus_1_to_minus_k):
        extremal_tails = [+1, -1]
        vals = []

        for tail in extremal_tails:
            x = self.theta0

            # include fixed w_-1...w_-k
            for i, wi in enumerate(w_minus_1_to_minus_k, start=1):
                x += self.theta[i] * wi

            # infinite exponential tail
            tail_contrib = self.theta.tail_sum(k)
            x += tail_contrib * tail

            p = self.q_logistic(x)
            vals.append(p if g == +1 else (1 - p))

        return min(vals)


    # --------------------------------------------------------
    # Compute a_k (the memory threshold)
    # --------------------------------------------------------

    def a_k(self, k):
        r_k = self.theta.tail_sum(k)  # analytic infinite tail
        C_plus = 0.5
        return max(0.0, 1 - 2*C_plus*r_k)


    # --------------------------------------------------------
    # Compute K_n from U_n  (eq. 3.4)
    # --------------------------------------------------------

    def compute_K(self, u):
        k = 0
        while True:
            if u < self.a_k(k):
                return k
            k += 1

    # --------------------------------------------------------
    # τ[n] definition (eq. 3.5)
    # --------------------------------------------------------

    def tau_of_n(self, U, window):
        s, t = window
        # Start from m = s and move downward
        for m in range(s, -10**18, -1):  # or while-loop; this just goes down indefinitely
            valid = True

            # Check all k in [m, t]
            for k in range(m, t + 1):
                if U[k] >= self.a_k(k - m):
                    valid = False
                    break

            if valid:
                return m
            else:
                return s - self.max_depth
                   
                    
                        

    def compute_user_impatience_bias(self, n, tau):
        """
        Compute the user impatience bias at time n.
        This is the probability that τ[n] is not a true regeneration time.
        """
        if tau <= n + self.max_depth:
            return 0.0
        else:
            return (self.a_k(self.max_depth))
    # --------------------------------------------------------
    # Construct X_τ ... X_n  (eq. 3.7)
    # --------------------------------------------------------

    def sample_interval(self, U, tau, window, debug=False):
        """
        Construct X_tau, ..., X_n with optional debugging output.
        Fully supports negative tau.
        """
        X = {}
        n = window[1]
        tau = int(tau)
        if debug:
            print("\n=== BEGIN INTERVAL SAMPLING ===")
            print(f"tau = {tau}, n = {window}")
            print("--------------------------------------------------------")

        for j in range(tau, n + 1):

            # =====================================================================
            # 1. Build past (X[j-1], X[j-2], ...), stopping at tau
            # =====================================================================
            past = []
            k = 1
            while True:
                idx = j - k
                if idx < tau:
                    break
                if idx not in X:
                    break
                past.append(X[idx])
                k += 1
            past = tuple(past)

            if debug:
                print(f"\n[j = {j}] Past = {past}")

            # =====================================================================
            # 2. Compute memory depth K_j
            # =====================================================================
            Kj = self.compute_K(U[j])

            if debug:
                print(f"  U[{j}] = {U[j]:.6f}")
                print(f"  K_j = {Kj}")

            # =====================================================================
            # 3. Build probability partition on [0,1]
            # =====================================================================
            intervals = []
            left = 0.0

            for k in range(Kj + 1):
                w = past[:k]
                if debug:
                    print(f"    k={k}, w={w}")

                for g in self.G:
                    akg = self.a_k_g_given_w(k, g, w)
                    intervals.append((left, left + akg, g))
                    if debug:
                        print(f"      interval: [{left:.6f}, {left+akg:.6f}) for g={g}")
                    left += akg

            # =====================================================================
            # 4. Choose X[j] according to partition
            # =====================================================================
            uj = U[j]
            assigned = False
            for L, R, g in intervals:
                if L <= uj < R:
                    X[j] = g
                    assigned = True
                    if debug:
                        print(f"  Selected g={g} because {L:.6f} <= {uj:.6f} < {R:.6f}")
                    break

            # Fallback for rare numerical issues
            if not assigned:
                X[j] = self.G[-1]
                if debug:
                    print(f"  WARNING: uj not in any interval; fallback to g={X[j]}")

        if debug:
            print("\n=== END INTERVAL SAMPLING ===\n")

        biases = []
        for key in X.keys():
            if key > 0:
                biases.append(self.compute_user_impatience_bias(key, tau))
        if debug:
            print(f"User impatience biases for sampled interval: {biases}")

        return X, np.array(biases).mean() if biases else 0.0


    # --------------------------------------------------------
    # Perfect Simulation of X_n (Section 3)
    # --------------------------------------------------------

    def perfect_sample(self, window=(3,5)):
        U = LazyU()
        j = window[1]
     
        # Step 1: Generate backward until τ[n] finite
        while True:
            tau_n = self.tau_of_n(U, window=window)
            print(f"At window={window}, τ[{window}]={tau_n}")
            if tau_n != float("-inf"):
                break
            j -= 1

        # Step 2: Construct X_τ,...,X_n
        X, biases = self.sample_interval(U, tau_n, window)
        window_index = list(range(window[0], window[1]+1))
        perfect_samples = []
        for i in window_index:
            perfect_samples.append(X[i])
        return perfect_samples, tau_n
    



class LazyU:
    def __init__(self):
        self.storage = {}   # only store accessed values

    def __getitem__(self, k):
        if k not in self.storage:
            # generate it once
            self.storage[k] = rng.uniform(0, 1)
        return self.storage[k]