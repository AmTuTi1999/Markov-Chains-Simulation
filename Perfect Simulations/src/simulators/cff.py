import math
# from src.simulators.perfect_simulator import PerfectSimulator
import functools

from tqdm import tqdm
from src.utils import LazyU

# ============================================================
# Binary Autoregressive Perfect Simulation (CFF 2002)
# ============================================================

class BinaryAutoregressiveSimulator:
    """
    Perfect simulation for the binary autoregressive process using
    the Comets–Fernández–Ferrari (2002) regenerative construction.

        P(X_0 = 1 | past) = q(theta0 + sum_k theta_k X_{-k})

    where q(x) = (1 + exp(-2x))^{-1}.
    """

    def __init__(
        self,
        theta0,
        theta_seq,
        alphabet=None,
        max_regen_search_depth=1000,
        show_progress=True,
    ):
        self.theta0 = float(theta0)
        self.theta = theta_seq
        self.G = alphabet if alphabet is not None else [-1, +1]
        self.max_depth = int(max_regen_search_depth)
        self.show_progress = show_progress

        assert set(self.G) == {-1, +1}, "Alphabet must be {-1, +1}"

    # --------------------------------------------------------
    # Logistic link and Lipschitz constant
    # --------------------------------------------------------

    @staticmethod
    def q_logistic(x):
        return 1.0 / (1.0 + math.exp(-2.0 * x))

    @staticmethod
    def q_prime(x):
        qx = BinaryAutoregressiveSimulator.q_logistic(x)
        return 2.0 * qx * (1.0 - qx)

    @property
    def C_plus(self):
        """
        Global Lipschitz constant of q.
        For q(x) = (1 + exp(-2x))^{-1}, max q'(x) = 1/2.
        """
        return 0.5

    # --------------------------------------------------------
    # Conditional probability
    # --------------------------------------------------------

    def conditional_P(self, g, history):
        """
        P(X_0 = g | history)
        """
        x = self.theta0
        for k, wk in enumerate(history, start=1):
            x += self.theta[k] * wk

        p = self.q_logistic(x)
        return p if g == +1 else (1.0 - p)

    # --------------------------------------------------------
    # a_k(g | w)
    # --------------------------------------------------------

    def a_k_g_given_w(self, k, g, w_minus_1_to_minus_k):
        """
        Lower bound a_k(g | w) using extremal tails.
        """
        vals = []

        for tail in (+1, -1):
            x = self.theta0

            # fixed part
            for i, wi in enumerate(w_minus_1_to_minus_k, start=1):
                x += self.theta[i] * wi

            # infinite tail bound
            x += self.theta.tail_sum(k) * tail

            p = self.q_logistic(x)
            vals.append(p if g == +1 else (1.0 - p))

        return min(vals)

    # --------------------------------------------------------
    # a_k (memory threshold)
    # --------------------------------------------------------

    @functools.lru_cache(None)
    def a_k(self, k):
        """
        a_k = 1 - 2 * C_plus * r_k
        """
        r_k = self.theta.tail_sum(k)
        return max(0.0, 1.0 - 2.0 * self.C_plus * r_k)

    # --------------------------------------------------------
    # Compute K_n from U_n
    # --------------------------------------------------------

    def compute_user_impatience_bias_given_limit(self):
        """
        Compute the user impatience bias given the limit.
        """
        return (1.0 - self.a_k(self.max_depth)) / self.a_k(self.max_depth)

    def conditional_lookback_expectation(self):
        """
        Compute E[K | K < max_depth].
        """
        expect_sum = 0.0

        for k in range(self.max_depth):
            ak = self.a_k(k)
            ak_next = self.a_k(k + 1)
            pk = ak_next - ak
            expect_sum += k * pk

        return expect_sum * self.a_k(self.max_depth) + 2 * (1 - self.a_k(self.max_depth)) * self.theta.tail_sum(self.max_depth)
    
    def compute_K(self, u):
        """
        Smallest k such that u < a_k(k).
        """
        for k in range(self.max_depth + 1):
            if u < self.a_k(k):
                return k
        return self.max_depth
    

    def analytic_lookback_bound(self,) -> float:
        """
        Analytic upper bound on expected lookback depth:

           

        where rho is the upper bound on the memory decay.
        """
        return self.theta.analytic_lookback_bound()

    # --------------------------------------------------------
    # τ[n] definition
    # --------------------------------------------------------

    def tau_of_n(self, U, window):
        """
        Compute regeneration time τ[n].
        """
        s, t = window
        lower_bound = s - self.max_depth

        for m in range(s, lower_bound - 1, -1):
            valid = True
            for k in range(m, t + 1):
                if U[k] >= self.a_k(k - m):
                    valid = False
                    break
            if valid:
                return m

        return lower_bound

    # --------------------------------------------------------
    # Construct X_tau ... X_n
    # --------------------------------------------------------

    def sample_interval(self, U, tau, window, debug=False):
        """
        Construct X_tau, ..., X_n.
        """
        X = {}
        n = window[1]
        tau = int(tau)

        iterator = range(tau, n + 1)
        if self.show_progress:
            iterator = tqdm(iterator, desc="Forward sampling")

        for j in iterator:

            # Build past
            past = []
            k = 1
            while True:
                idx = j - k
                if idx < tau or idx not in X:
                    break
                past.append(X[idx])
                k += 1
            past = tuple(past)

            # Memory depth
            Kj = self.compute_K(U[j])

            # Probability partition
            intervals = []
            left = 0.0

            for k in range(Kj + 1):
                w = past[:k]
                for g in self.G:
                    akg = self.a_k_g_given_w(k, g, w)
                    akg = min(akg, 1.0 - left)  # numerical safety
                    intervals.append((left, left + akg, g))
                    left += akg

            # Sample
            uj = U[j]
            for L, R, g in intervals:
                if L <= uj < R:
                    X[j] = g
                    break
            else:
                # fallback
                X[j] = self.G[-1]

        return X

    # --------------------------------------------------------
    # Perfect simulation
    # --------------------------------------------------------

    def perfect_sample(self, window=(0, 0)):
        """
        Perfectly sample X_window[0] ... X_window[1].
        """
        U = LazyU()
        s, t = window

        iterator = range(t, s - self.max_depth - 1, -1)
        if self.show_progress:
            iterator = tqdm(iterator, desc="Backward search")

        tau_n = None
        for _ in iterator:
            tau_n = self.tau_of_n(U, window)
            if tau_n > s - self.max_depth:
                break

        X = self.sample_interval(U, tau_n, window)
        samples = [X[i] for i in range(s, t + 1)]
        return samples, tau_n

    



