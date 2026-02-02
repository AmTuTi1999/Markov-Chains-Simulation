import math
import functools
import numpy as np
from tqdm import tqdm
from src.utils import LazyU
from typing import Literal, Optional, Dict, Tuple

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

    @property
    def alphabet_size(self):
        """Size of the alphabet |A|."""
        return len(self.G)

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
    # a_k (memory threshold / minorant)
    # --------------------------------------------------------

    @functools.lru_cache(None)
    def a_k(self, k):
        """
        Minorant α_k: probability that lookback depth ≤ k.
        
        Formula: α_k = max{0, 1 - 2·C_+·r_k}
        where r_k = sum_{m>k} |theta_m| is the tail sum.
        """
        r_k = self.theta.tail_sum(k)
        return max(0.0, 1.0 - 2.0 * self.C_plus * r_k)

    # ========================================================
    # NON-TRUNCATED PERFECT SIMULATION BOUNDS
    # ========================================================

    def non_truncated_expectation_analytical(self) -> float:
        """
        Compute E[L_n] for NON-TRUNCATED perfect simulation.
        
        This is the "true" expected lookback depth when you search 
        backwards until you actually find a regeneration time (no 
        artificial truncation at S).
        
        Formula:
            E[L_n] = sum_{k=0}^∞ k·P(L_n = k)
                   = sum_{k=0}^∞ k·(α_k - α_{k-1})
                   = sum_{k=0}^∞ P(L_n > k)
                   = sum_{k=0}^∞ (1 - α_k)
        
        This is computed by summing until α_k → 1 (convergence).
        
        Returns:
            Expected lookback depth (non-truncated)
        """
        expect_sum = 0.0
        tolerance = 1e-10
        max_iterations = 100000
        
        for k in range(max_iterations):
            prob_exceed_k = 1.0 - self.a_k(k)
            expect_sum += prob_exceed_k
            
            # Stop when α_k ≈ 1 (convergence)
            if prob_exceed_k < tolerance:
                break
        else:
            # Didn't converge - estimate remaining tail
            # E[(L_n - K)_+] for K = max_iterations
            remaining_tail = self.tail_expectation_general_from_k(max_iterations)
            expect_sum += remaining_tail
            print(f"Warning: Non-truncated sum did not converge after {max_iterations} "
                  f"iterations. Remaining tail estimated as {remaining_tail:.2e}")
        
        return expect_sum

    def tail_expectation_general_from_k(self, k: int) -> float:
        """
        Estimate sum_{j>k} (1 - α_j) using θ tail sum.
        
        Used for non-truncated expectation when direct summation doesn't converge.
        """
        return self.alphabet_size * self.theta.tail_sum(k)

    def non_truncated_bound_exponential(self, C: float, rho: float) -> float:
        """
        Analytical bound on E[L_n] for NON-TRUNCATED perfect simulation
        under exponential decay.
        
        Bound: E[L_n] = sum_{k≥0} (1 - α_k) 
                      ≤ sum_{k≥0} |A|·β_k  
                      ≤ |A|·C·sum_{k≥0} ρ^k
                      = |A|·C/(1-ρ)
        
        This is Theorem 5.1.22 case (i).
        
        Args:
            C: Upper bound constant
            rho: Decay rate (0 < rho < 1)
            
        Returns:
            Upper bound on E[L_n]
        """
        if rho >= 1.0:
            return float('inf')
        
        return self.alphabet_size * C / (1.0 - rho)

    def non_truncated_bound_polynomial(self, C: float, p: float) -> float:
        """
        Analytical bound on E[L_n] for NON-TRUNCATED perfect simulation
        under polynomial decay.
        
        Bound: E[L_n] ≤ |A|·C·sum_{k≥0} k^(-p)
                      = |A|·C·(1 + ζ(p))
        
        where ζ(p) is the Riemann zeta function.
        
        This is Theorem 5.1.22 case (ii).
        
        Args:
            C: Upper bound constant
            p: Decay exponent (p > 1)
            
        Returns:
            Upper bound on E[L_n]
        """
        if p <= 1.0:
            return float('inf')
        
        from scipy.special import zeta
        return self.alphabet_size * C * (1.0 + zeta(p))

    def non_truncated_analytical_bound(
        self,
        decay_type: Literal['exponential', 'polynomial', 'general'] = 'general',
        C: Optional[float] = None,
        param: Optional[float] = None,
        compute_exact: bool = False
    ) -> Dict[str, float]:
        """
        Complete analytical bound for NON-TRUNCATED perfect simulation.
        
        Args:
            decay_type: Type of decay
            C: Constant for parametric bounds
            param: rho (exponential) or p (polynomial)
            compute_exact: If True, compute exact E[L_n] by summation
            
        Returns:
            Dictionary with bound and exact value (if computed)
        """
        # Upper bound from theorem
        if decay_type == 'exponential':
            if C is None or param is None:
                raise ValueError("C and param (rho) required")
            theoretical_bound = self.non_truncated_bound_exponential(C, param)
        elif decay_type == 'polynomial':
            if C is None or param is None:
                raise ValueError("C and param (p) required")
            theoretical_bound = self.non_truncated_bound_polynomial(C, param)
        else:  # general
            theoretical_bound = None
        
        result = {
            'theoretical_bound': theoretical_bound,
            'decay_type': decay_type,
            'method': 'non-truncated analytical bound',
            'source': 'Theorem 5.1.22'
        }
        
        # Optionally compute exact value
        if compute_exact:
            exact_value = self.non_truncated_expectation_analytical()
            result['exact_value'] = exact_value

        
        return result

    # ========================================================
    # TRUNCATED PERFECT SIMULATION BOUNDS (from before)
    # ========================================================

    def truncated_expectation_analytical(self) -> float:
        """
        Compute E[ψ_S(L_n)] = E[min(L_n, S)] analytically using minorants.
        
        This is EXACT μ_S from theory (no Monte Carlo error).
        
        Formula:
            μ_S = sum_{k=0}^{S-1} k·P(L_n = k) + S·P(L_n ≥ S)
                = sum_{k=0}^{S-1} k·(α_k - α_{k-1}) + S·(1 - α_S)
        
        Returns:
            Exact truncated expectation
        """
        S = self.max_depth
        expect_sum = 0.0
        
        # Sum over k = 0, 1, ..., S-1
        for k in range(S):
            if k == 0:
                pk = self.a_k(0)
            else:
                pk = self.a_k(k) - self.a_k(k - 1)
            
            expect_sum += k * pk
        
        # Add contribution from truncated tail: S·P(L_n ≥ S)
        prob_exceed_S = 1.0 - self.a_k(S)
        expect_sum += S * prob_exceed_S
        
        return expect_sum

    def tail_expectation_exponential(self, C: float, rho: float) -> float:
        """
        Compute E[(L_n - S)_+] for exponential decay regime.
        
        Bound: E[(L_n - S)_+] ≤ |A|·C·ρ^(S+1)/(1-ρ)
        
        Args:
            C: Upper bound constant on continuity rate
            rho: Decay rate (0 < rho < 1)
            
        Returns:
            Upper bound on tail expectation
        """
        if rho >= 1.0:
            return float('inf')
        
        S = self.max_depth
        bound = self.alphabet_size * C * (rho ** (S + 1)) / (1.0 - rho)
        return bound

    def tail_expectation_polynomial(self, C: float, p: float) -> float:
        """
        Compute E[(L_n - S)_+] for polynomial decay regime.
        
        Bound: E[(L_n - S)_+] ≤ |A|·C·S^(1-p)/(p-1)
        
        Args:
            C: Upper bound constant on continuity rate
            p: Decay exponent (p > 1)
            
        Returns:
            Upper bound on tail expectation
        """
        if p <= 1.0:
            return float('inf')
        
        S = self.max_depth
        bound = self.alphabet_size * C * (S ** (1.0 - p)) / (p - 1.0)
        return bound

    def tail_expectation_general(self) -> float:
        """
        Compute E[(L_n - S)_+] using general formula.
        
        Bound: E[(L_n - S)_+] ≤ |A|·sum_{k>S} β_k ≤ |A|·tail_sum(S)
        
        Returns:
            Upper bound on tail expectation
        """
        S = self.max_depth
        return self.alphabet_size * self.theta.tail_sum(S)

    def truncated_analytical_bound(
        self, 
        decay_type: Literal['exponential', 'polynomial', 'general'] = 'general',
        C: Optional[float] = None,
        param: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Compute COMPLETE analytical bound on E[L_n] for TRUNCATED simulation.
        
        NO MONTE CARLO ERROR - computed directly from minorants.
        
        Formula:
            E[L_n] ≤ μ_S + E[(L_n - S)_+]
        
        Args:
            decay_type: 'exponential', 'polynomial', or 'general'
            C: Constant for parametric bounds (required for exp/poly)
            param: rho for exponential, p for polynomial
            
        Returns:
            Dictionary with bound components and metadata
        """
        # Compute μ_S analytically (EXACT)
        mu_S = self.truncated_expectation_analytical()
        
        # Compute tail bound
        if decay_type == 'exponential':
            if C is None or param is None:
                raise ValueError("C and param (rho) required for exponential decay")
            tail = self.tail_expectation_exponential(C, param)
        elif decay_type == 'polynomial':
            if C is None or param is None:
                raise ValueError("C and param (p) required for polynomial decay")
            tail = self.tail_expectation_polynomial(C, param)
        else:  # general
            tail = self.tail_expectation_general()
        
        total = mu_S + tail
        
        return {
            'mu_S_analytical': mu_S,
            'tail_bound': tail,
            'total_bound': total,
            'truncation_index': self.max_depth,
            'decay_type': decay_type,
            'method': 'truncated analytical (no MC error)',
            'prob_exceed_S': 1.0 - self.a_k(self.max_depth)
        }

    # ========================================================
    # UNIFIED INTERFACE
    # ========================================================

    def analytical_lookback_bound(
        self,
        truncated: bool = True,
        **kwargs
    ) -> Dict[str, float]:
        """
        Unified interface for analytical lookback bounds.
        
        Args:
            truncated: If True, use truncated bound (user-imposed limit S)
                      If False, use non-truncated bound (true perfect simulation)
            **kwargs: Additional arguments passed to specific bound methods
            
        Returns:
            Dictionary with bound information
        """
        if truncated:
            return self.truncated_analytical_bound(**kwargs)
        else:
            return self.non_truncated_analytical_bound(**kwargs)

    # --------------------------------------------------------
    # Empirical Validation (for both truncated and non-truncated)
    # --------------------------------------------------------

    def empirical_lookback_samples(
        self, 
        num_samples: int = 10000,
        truncated: bool = True
    ) -> np.ndarray:
        """
        Generate empirical samples of lookback depth L_n.
        
        Args:
            num_samples: Number of independent samples
            truncated: If True, truncate at max_depth; if False, search until regeneration
            
        Returns:
            Array of lookback depths
        """
        samples = []
        
        iterator = range(num_samples)
        if self.show_progress:
            iterator = tqdm(iterator, desc="Sampling lookback depths")
        
        for _ in iterator:
            U = LazyU()
            
            # Find smallest k such that U[0] < a_k
            k = 0
            max_search = self.max_depth if truncated else 100000
            
            while k <= max_search:
                if U[0] < self.a_k(k):
                    break
                k += 1
            
            if truncated:
                samples.append(min(k, self.max_depth))
            else:
                samples.append(k)
        
        return np.array(samples)

    def validate_analytical_bound(
        self,
        num_samples: int = 10000,
        truncated: bool = True,
        **kwargs
    ) -> Dict[str, float]:
        """
        Validate analytical bound against empirical samples.
        
        Args:
            num_samples: Number of samples for validation
            truncated: Whether to use truncated or non-truncated version
            **kwargs: Arguments for analytical_lookback_bound
            
        Returns:
            Comparison dictionary
        """
        # Get analytical bound
        analytical = self.analytical_lookback_bound(truncated=truncated, **kwargs)
        
        # Generate empirical samples
        samples = self.empirical_lookback_samples(num_samples, truncated=truncated)
        
        # Statistics
        empirical_mean = np.mean(samples)
        empirical_std = np.std(samples)
        empirical_std_error = empirical_std / np.sqrt(num_samples)
        
        # Comparison
        if truncated:
            analytical_value = analytical['mu_S_analytical']
        else:
            analytical_value = analytical.get('exact_value', analytical['theoretical_bound'])
        
        discrepancy = analytical_value - empirical_mean
        relative_discrepancy = discrepancy / analytical_value if analytical_value > 0 else 0

        result = analytical.copy()
        result['tightness'] = empirical_mean / analytical_value if analytical_value > 0 else float('inf')

        return {
            **result,
            'empirical_mean': empirical_mean,
            'empirical_std': empirical_std,
            'empirical_std_error': empirical_std_error,
            'discrepancy': discrepancy,
            'relative_discrepancy': relative_discrepancy,
            'num_samples': num_samples,
            'truncated': truncated
        }

    # --------------------------------------------------------
    # Legacy methods and perfect sampling (unchanged)
    # --------------------------------------------------------

    def compute_user_impatience_bias_given_limit(self) -> float:
        """User impatience bias for truncated simulation."""
        prob_exceed = 1.0 - self.a_k(self.max_depth)
        prob_within = self.a_k(self.max_depth)
        return prob_exceed / prob_within if prob_within > 0 else float('inf')

    def compute_K(self, u):
        """Smallest k such that u < a_k."""
        for k in range(self.max_depth + 1):
            if u < self.a_k(k):
                return k
        return self.max_depth

    def tau_of_n(self, U, window):
        """Compute regeneration time τ[n]."""
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

    def sample_interval(self, U, tau, window, debug=False):
        """Construct X_tau, ..., X_n."""
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
                    akg = min(akg, 1.0 - left)
                    intervals.append((left, left + akg, g))
                    left += akg

            # Sample
            uj = U[j]
            for L, R, g in intervals:
                if L <= uj < R:
                    X[j] = g
                    break
            else:
                X[j] = self.G[-1]

        return X

    def perfect_sample(self, window: Tuple[int, int] = (0, 0)):
        """Perfectly sample X_window[0] ... X_window[1]."""
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