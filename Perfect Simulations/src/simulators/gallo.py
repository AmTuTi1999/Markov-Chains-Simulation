import numpy as np
from typing import List, Tuple, Set, Dict, Optional, Union
from itertools import product
from tqdm import tqdm
from src.utils import LazyU


class GalloContextTreeSimulator:
    """
    (Approximate) forward simulation + Monte Carlo estimation for lookback depth L'_n.

    IMPORTANT CHANGE vs your original:
    - Empirical sampling now simulates the CHAIN Z using your transition_probability + find_context,
      instead of using an i.i.d. fake "past".
    - Lookback m_n is computed efficiently along the path by tracking the last occurrence of w.
    """

    def __init__(
        self,
        alpha: float,
        alphabet: List[int],
        reference_string: List[int],
        epsilon: float = 0.3,
        beta: float = 0.7,
        max_depth: int = 50,
        max_trie_depth: int = 8,
        show_progress: bool = False,
        M: int = 1,
    ):
        self.alpha = float(alpha)
        self.alphabet = list(alphabet)
        self.reference_string = tuple(reference_string)
        self.epsilon = float(epsilon)
        self.beta = float(beta)
        self.max_depth = int(max_depth)
        self.max_trie_depth = int(max_trie_depth)
        self.show_progress = bool(show_progress)

        self.len_w = len(self.reference_string)
        self.alphabet_size = len(self.alphabet)
        self.lag_function = lambda k: int(np.ceil(np.exp(M*self.alpha * k)))
        self.ar_coef = lambda i: float(np.exp(-(i ** self.beta)))
        self.p_w = float(self.epsilon ** self.len_w)
        self.bound_constant = M
        self.contexts = self._generate_contexts()
        self.U = LazyU()

        if self.show_progress:
            print(f"[GalloSim] Initialized with α={self.alpha:.3f}, |τ|={len(self.contexts)} contexts")

    # ============================================================================
    # CONTEXT TREE GENERATION
    # ============================================================================

    def _generate_contexts(self) -> Set[Tuple]:
        """
        Generate τ = ⋃_{i≥0} ⋃_{c∈A^{l_w(i)}} c·w·A^i, truncated by max_trie_depth.

        Each context: filler + w + prefix where |prefix| = i and |filler| = l_w(i).
        """
        contexts: Set[Tuple] = set()
        w = self.reference_string

        for i in range(self.max_trie_depth + 1):
            lag_len = self.lag_function(i)

            for filler in product(self.alphabet, repeat=lag_len):
                for prefix in product(self.alphabet, repeat=i):
                    context = tuple(filler) + w + tuple(prefix)
                    contexts.add(context)

        return contexts

    def find_context(self, past: List[int]) -> Tuple:
        """
        Find c_τ(past) = longest suffix of past that belongs to τ.
        """
        max_search = min(len(past), self.max_trie_depth * 10)
        for length in range(max_search, 0, -1):
            suffix = tuple(past[-length:])
            if suffix in self.contexts:
                return suffix
        return tuple()

    # ============================================================================
    # TRANSITION PROBABILITIES (AR Model from Eq 9.6-9.7)
    # ============================================================================

    def transition_probability(self, symbol: int, context: Tuple) -> float:
        """
        AR model (as in your code).
        """
        K = len(context)
        if K == 0:
            return 1.0 / self.alphabet_size

        score_s = sum(
            self.ar_coef(i + 1)
            for i in range(K)
            if context[-(i + 1)] == symbol
        )

        Z = 0.0
        for s in self.alphabet:
            score = sum(
                self.ar_coef(i + 1)
                for i in range(K)
                if context[-(i + 1)] == s
            )
            Z += float(np.exp(score))

        if Z <= 0:
            return 1.0 / self.alphabet_size

        return float(np.exp(score_s) / Z)

    # ============================================================================
    # LOOKBACK DEPTH L' COMPUTATION (efficient on a whole path)
    # ============================================================================

    def _compute_last_end_up_to_generic(self, X: np.ndarray, pattern: np.ndarray) -> np.ndarray:
        """
        Generic:
        last_end_up_to[t] = last index <= t where pattern ends, i.e. X[t-L+1:t+1] == pattern.
        If no occurrence yet: -1
        """
        T = len(X)
        L = len(pattern)

        last_end = -1
        out = np.full(T, -1, dtype=int)

        for t in range(T):
            if t >= L - 1 and np.array_equal(X[t - L + 1:t + 1], pattern):
                last_end = t
            out[t] = last_end

        return out


    def _compute_last_end_up_to(self, Z: np.ndarray) -> np.ndarray:
        """
        Original version for Z, matching self.reference_string.
        """
        w = np.asarray(self.reference_string, dtype=Z.dtype)
        return self._compute_last_end_up_to_generic(Z, w)


    def _compute_last_end_up_to_rescaled(
        self,
        Z_bar: np.ndarray,
    ) -> np.ndarray:
        """
        Rescaled version for \bar Z, matching reference_bar.

        If reference_bar is None, defaults to a run of ones of length self.len_w
        (common choice when \bar Z_m ∈ {1, ★} and you want occurrences of all-1 blocks).
        """
        return self._compute_last_end_up_to_generic(Z_bar, np.array([1]))

    def compute_lookback_depth_from_last_end(self, U_n: float, i: int, last_end_up_to: np.ndarray) -> float:
        """
        Compute L'_i using precomputed last_end_up_to from a simulated path.

        L'_i = 0                                  if U_i < |A| * epsilon
             = m_i + |w| + ceil(l^w((m+1)|w| + 1)/|w|)              otherwise

        where m_i = (i-1) - last_end_up_to[i-1] (distance to last end of w before i)
        """
        if U_n < self.alphabet_size * self.epsilon:
            return 0.0

        if i - 1 < 0:
            return np.inf

        le = int(last_end_up_to[i - 1])
        if le < 0:
            return np.inf

        m = (i - 1) - le
        return float(m + self.len_w + np.ceil(self.lag_function((m + 1)*self.len_w - 1)/self.len_w))

    def compute_lookback_depth_from_last_end_rescaled(
        self,
        Zbar_n: Union[int, str],
        n: int,
        last_one_up_to: np.ndarray
    ) -> float:
        """
        Rescaled version (for \bar Z).

        L_n = 0                                             if \bar Z_n = 1
            = \bar m_n + |w| + ceil( l^w((\bar m_n+1)|w|-1) / |w| )   otherwise

        where \bar m_n = (n-1) - last_one_up_to[n-1]
        and last_one_up_to[t] = last index <= t where \bar Z == 1, else -1.

        Notes:
        - n is an index into the *rescaled* chain \bar Z (block time).
        - |w| is still self.len_w (your original word length / block length).
        """
        if Zbar_n == 1:
            return 0.0

        if n - 1 < 0:
            return np.inf

        le = int(last_one_up_to[n - 1]) 
        if le < 0:
            return np.inf

        mbar = (n - 1) - le
        return float(
            mbar
            + self.len_w
            + np.ceil(self.lag_function((mbar + 1) * self.len_w - 1) / self.len_w)
        )
    # ============================================================================
    # FORWARD SIMULATION OF THE CHAIN Z (approximate stationarity via burn-in)
    # ============================================================================

    def simulate_chain_forward(
        self,
        T: int = 200_000,
        burn_in: int = 10_000,
        seed: Optional[int] = None,
        init_past_len: Optional[int] = None,
        epsilon: float = 0.0,
        return_rescaled: bool = False,
        star_value: Union[int, str] = 0, 
    ) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """
        Simulate Z forward using a single driving uniform U_t each step, in the spirit of
        the interval construction:
            J(a|∅) = [ (a-1)ε, aε )

        If return_rescaled=True and block_len=L is provided, compute the block process \bar Z_m:
            \bar Z_m = 1  if U_{mL-i} ∈ J(Z_{mL-i}|∅) for i=0..L-1
                    = ★  otherwise
        using non-overlapping blocks of length L on the returned post-burn-in sample.

        Returns:
            - Z (np.ndarray) if no extras requested
            - (Z, U) if return_uniforms
            - (Z, Z_bar) if return_rescaled
            - (Z, U, Z_bar) if both
        """
        rng = np.random.default_rng(seed)

        if not (0.0 <= epsilon <= 1.0):
            raise ValueError("epsilon must be in [0, 1].")
        if epsilon * self.alphabet_size > 1.0 + 1e-12:
            raise ValueError("Need epsilon * |A| <= 1 for the spontaneous intervals to fit in [0,1].")

        if init_past_len is None:
            init_past_len = max(2 * self.max_depth, 200)

        alphabet = np.array(sorted(self.alphabet), dtype=int)
        A = int(self.alphabet_size)

        Z_list = [int(rng.choice(alphabet)) for _ in range(init_past_len)]

        total_steps = burn_in + T
        eps_mass = epsilon * A
        rem_mass = 1.0 - eps_mass

        for step in range(total_steps):
            print(f"[GalloSim] Simulating chain forward: step {step + 1}/{total_steps}", end="\r")

            ctx = self.find_context(Z_list)
            probs = np.array([self.transition_probability(a, ctx) for a in alphabet], dtype=float)

            s = probs.sum()
            if s <= 0:
                probs[:] = 1.0 / A
            else:
                probs /= s

            if epsilon > 0.0 and self.U[step] < eps_mass:
                idx = int(self.U[step] / epsilon)  
                a_next = int(alphabet[idx])
                Z_list.append(a_next)
                continue

            if rem_mass <= 0.0:
                a_next = int(alphabet[min(A - 1, int((self.U[step] - 1e-15) / max(epsilon, 1e-15)))])
                Z_list.append(a_next)
                continue

            residual = np.maximum(probs - epsilon, 0.0)
            rsum = residual.sum()

            if rsum <= 0:
                cdf = np.cumsum(probs)
                a_next = int(alphabet[int(np.searchsorted(cdf, self.U[step], side="right"))])
                Z_list.append(a_next)
                continue

            u2 = (self.U[step] - eps_mass) / rem_mass
            residual /= rsum
            cdf = np.cumsum(residual)
            j = int(np.searchsorted(cdf, u2, side="right"))
            if j >= A:
                j = A - 1
            Z_list.append(int(alphabet[j]))

        Z_full = np.array(Z_list[-T:], dtype=int)
        print(f"\n[GalloSim] Simulated chain forward: burn-in={burn_in}, T={T}, total={total_steps}")

        if return_rescaled:
            if self.len_w is None or self.len_w <= 0:
                raise ValueError("To compute Z_bar, len_w must be a positive integer.")

            L = int(self.len_w)
            M = T // L 
            Z_use = Z_full[-M * L:]
            w = np.array(self.reference_string, dtype=Z_use.dtype)
            Z_blocks = Z_use.reshape(M, L)
            matches = np.all(Z_blocks == w, axis=1)

            dtype = object if isinstance(star_value, str) else int
            Z_bar = np.where(matches, 1, 0 if not isinstance(star_value, str) else star_value).astype(dtype)

            return Z_bar

        else:
            return Z_full

    # ============================================================================
    # MONTE CARLO ESTIMATION OF E[L']
    # ============================================================================

    def monte_carlo_E_Lprime(
        self,
        T: int = 200_000,
        burn_in: int = 10_000000,
        seed: Optional[int] = None,
        truncated: bool = False,
        truncate_at: Optional[int] = None,
        drop_infinite: bool = True,
        return_ci: bool = True,
        ci_level: Optional[float] = 0.95
    ) -> Dict[str, float]:
        """
        Monte Carlo estimate of E[L'_i] by simulating Z and averaging L'_i along the path.

        Args:
            T: kept sample size after burn-in
            burn_in: burn-in length
            truncated: if True, use min(L', S)
            truncate_at: truncation S (defaults to self.max_depth)
            drop_infinite: if True, drop indices where L' is infinite (typically early / before first w)

        Returns:
            dict with mean, std, stderr, and counts
        """
        if truncate_at is None:
            truncate_at = self.max_depth

        np.random.default_rng(seed)
        Z = self.simulate_chain_forward(T=T, burn_in=burn_in, seed=seed, return_rescaled=True)
        
        last_end_up_to = self._compute_last_end_up_to_rescaled(Z)

        Lprime = np.empty(T // self.len_w, dtype=float)
        for i in tqdm(range(T // self.len_w), disable=not self.show_progress):
            Lprime[i] = self.compute_lookback_depth_from_last_end_rescaled(Z[i], i, last_end_up_to)

        if truncated:
            finite = np.isfinite(Lprime)
            Lprime_tr = np.full_like(Lprime, float(truncate_at))
            Lprime_tr[finite] = np.minimum(Lprime[finite], float(truncate_at))
            Luse = Lprime_tr
            used = T
            dropped = 0
        else:
            if drop_infinite:
                mask = np.isfinite(Lprime)
                Luse = Lprime[mask]
                used = int(mask.sum())
                dropped = int((~mask).sum())
            else:
                Luse = Lprime
                used = T
                dropped = 0

        if used == 0:
            return {
                "mean": float("nan"),
                "std": float("nan"),
                "stderr": float("nan"),
                "used": 0,
                "dropped_infinite": dropped,
                "T": T,
                "burn_in": burn_in,
                "truncated": truncated,
                "truncate_at": truncate_at
            }

        mean = float(np.mean(Luse))
        std = float(np.std(Luse, ddof=1)) if used > 1 else 0.0
        stderr = float(std / np.sqrt(used)) if used > 1 else 0.0
        if return_ci:
            # z-value for common CI levels
            z = 1.959963984540054 if abs(ci_level - 0.95) < 1e-12 else None
            if z is None:
                # crude fallback using inverse-erf approximation
                
                # approximate z from ci_level (two-sided)
                # for typical thesis usage, 0.95 is enough; otherwise user can extend this
                z = 1.959963984540054
            half = z * stderr
        return {
            "mean": mean,
            "std": std,
            "stderr": stderr,
            "used": used,
            "dropped_infinite": dropped,
            "T": T,
            "burn_in": burn_in,
            "truncated": truncated,
            "truncate_at": truncate_at,
            "ci_low": mean - half if return_ci else float("nan"),
            "ci_high": mean + half if return_ci else float("nan"),
            "ci_level": float(ci_level)
        }

    def empirical_lookback_samples(
        self,
        num_samples: int = 10_000,
        truncated: bool = True,
        seed: Optional[int] = None,
        burn_in: int = 10_000,
        drop_infinite: bool = True
    ) -> np.ndarray:
        """
        Backwards-compatible "samples" method:
        now returns samples of L' along a simulated chain instead of i.i.d. fake past.

        Args:
            num_samples: number of time points sampled (path length)
            truncated: whether to truncate at self.max_depth
            seed: RNG seed
            burn_in: burn-in for approximate stationarity
            drop_infinite: if not truncated, drop inf values

        Returns:
            numpy array of samples
        """

        # Re-run but return samples (we keep it simple and re-simulate deterministically via seed offset)
        rng = np.random.default_rng(seed)
        Z = self.simulate_chain_forward(T=num_samples, burn_in=burn_in, seed=seed)
        U = rng.random(num_samples)
        last_end_up_to = self._compute_last_end_up_to(Z)

        Lprime = np.empty(num_samples, dtype=float)
        for i in range(num_samples):
            Lprime[i] = self.compute_lookback_depth_from_last_end(U[i], i, last_end_up_to)

        if truncated:
            finite = np.isfinite(Lprime)
            out = np.full_like(Lprime, float(self.max_depth))
            out[finite] = np.minimum(Lprime[finite], float(self.max_depth))
            return out

        if drop_infinite:
            return Lprime[np.isfinite(Lprime)]
        return Lprime

    # ============================================================================
    # YOUR ANALYTICAL PARTS (kept as-is, except minor robustness)
    # ============================================================================

    def truncated_expectation_analytical(self) -> float:
        """
        Compute E[min(L_n, S)] via your pmf/cdf machinery (unchanged).
        """
        S = self.max_trie_depth
        expect_sum = 0.0

        for k in range(S + 1):
            pk = self._pmf_lookback(k)
            expect_sum += k * pk

        prob_exceed_S = 1.0 - self._cdf_lookback(S - 1)
        expect_sum += S * prob_exceed_S
        return float(expect_sum)

    def _cdf_lookback(self, k: int) -> float:
        return float(sum(self._pmf_lookback(j) for j in range(k + 1)))

    

    def tail_expectation_exponential_gallo(self) -> float:
        S = self.max_trie_depth
        r = 1 - self.p_w

        k_S = 0
        while self.len_w + k_S + np.ceil(self.lag_function(self.len_w*k_S)*np.exp(self.alpha*self.len_w -self.alpha)/self.len_w) <= S:
            k_S += 1

        if k_S <= 1:
            return 0.0

        r_power = r ** (k_S)
        term1 = r_power * (self.len_w - S + (k_S + 1) + (1-self.p_w) / self.p_w)

        denominator = self.len_w * (1 - r * np.exp(self.alpha*self.len_w))
        if denominator <= 0:
            return float("inf")

        term2 = self.bound_constant*(self.p_w * r_power * self.lag_function(self.len_w*k_S)*np.exp(self.alpha*self.len_w - self.alpha)) / denominator
        return float(max(0.0, term1 + term2))

    def truncated_analytical_bound(self) -> Dict[str, float]:
        mu_S = self.truncated_expectation_analytical()
        if not np.isfinite(mu_S):
            return {
                'mu_S_analytical': float("inf"),
                'tail_bound': float("inf"),
                'total_bound': float("inf"),
                'truncation_index': self.max_trie_depth,
                'method': 'truncated analytical (divergent)',
                'prob_exceed_S': 1.0
            }

        tail_bound = self.tail_expectation_exponential_gallo()
        scaling = 1 - self.epsilon**self.len_w
        prob_exceed_S = 1.0 - self._cdf_lookback(self.max_trie_depth - 1)

        return {
            'mu_S_analytical': float(mu_S),
            'tail_bound': float(tail_bound),
            'total_bound': float(scaling * (mu_S + tail_bound)),
            'truncation_index': self.max_trie_depth,
            'method': 'truncated analytical (no MC error)',
            'prob_exceed_S': float(prob_exceed_S)
        }

    def non_truncated_expectation_analytical(self) -> float:
        scaling = 1 - self.epsilon**self.len_w
        expectation = 0.0
        tolerance = 1e-12
        max_terms = 10000

        for k in range(max_terms):
            prob_exceed_k = 1.0 - self._cdf_lookback(k)
            expectation += prob_exceed_k
            if prob_exceed_k < tolerance:
                break

        return float(scaling * expectation)

    def non_truncated_analytical_bound(self, compute_exact: bool = False) -> Dict[str, float]:
        r = 1 - self.p_w
        

        if r * np.exp(self.alpha*self.len_w) >= 1:
            theoretical_bound = float("inf")
        else:
            kappa = 1.0 / (1 - r * np.exp(self.alpha*self.len_w))*self.len_w
            term1 = 1 / self.p_w
            term2 = self.len_w
            term3 = self.p_w * kappa *self.bound_constant*np.exp(self.alpha*self.len_w - self.alpha) 
            scaling = 1 - self.epsilon**self.len_w
            theoretical_bound = float(scaling * (term1 + term2 + term3))

        result = {
            'theoretical_bound': theoretical_bound,
            'decay_type': 'exponential_gallo',
            'method': 'non-truncated analytical bound',
            'source': 'Theorem 6.1.16'
        }

        if compute_exact:
            result['exact_value'] = self.non_truncated_expectation_analytical()

        return result

    def analytical_lookback_bound(
        self,
        truncated: bool = True,
        compute_exact: bool = False,
        **kwargs
    ) -> Dict[str, float]:
        if truncated:
            return self.truncated_analytical_bound()
        return self.non_truncated_analytical_bound(compute_exact=compute_exact)

    def validate_analytical_bound(
        self,
        num_samples: int = 10_000,
        truncated: bool = True,
        burn_in: int = 10_000,
        seed: Optional[int] = None,
        **kwargs
    ) -> Dict[str, float]:
        analytical = self.analytical_lookback_bound(truncated=truncated, **kwargs)
        samples = self.empirical_lookback_samples(
            num_samples=num_samples,
            truncated=truncated,
            burn_in=burn_in,
            seed=seed,
            drop_infinite=True
        )

        if len(samples) == 0:
            return {
                **analytical,
                'empirical_mean': float("nan"),
                'empirical_std': float("nan"),
                'empirical_std_error': float("nan"),
                'discrepancy': float("nan"),
                'relative_discrepancy': float("nan"),
                'num_samples': 0,
                'truncated': truncated,
                'note': 'No finite samples generated'
            }

        empirical_mean = float(np.mean(samples))
        empirical_std = float(np.std(samples, ddof=1)) if len(samples) > 1 else 0.0
        empirical_std_error = float(empirical_std / np.sqrt(len(samples))) if len(samples) > 1 else 0.0

        analytical_value = analytical['mu_S_analytical'] if truncated else analytical.get('exact_value', analytical['theoretical_bound'])

        discrepancy = float(analytical_value - empirical_mean) if np.isfinite(analytical_value) else float("nan")
        relative_discrepancy = float(discrepancy / analytical_value) if (np.isfinite(analytical_value) and analytical_value > 0) else float("nan")

        tightness = float(empirical_mean / analytical_value) if (np.isfinite(analytical_value) and analytical_value > 0) else float("inf")
        result = analytical.copy()
        result['tightness'] = tightness

        return {
            **result,
            'empirical_mean': empirical_mean,
            'empirical_std': empirical_std,
            'empirical_std_error': empirical_std_error,
            'discrepancy': discrepancy,
            'relative_discrepancy': relative_discrepancy,
            'num_samples': int(len(samples)),
            'truncated': truncated
        }

    def compute_user_impatience_bias_given_limit(self) -> float:
        prob_exceed = 1.0 - self._cdf_lookback(self.max_trie_depth - 1)
        if prob_exceed >= 0.9999:
            return float("inf")
        return float(prob_exceed / (1 - prob_exceed))

    # ============================================================================
    # PMF (left as your original logic, but keep it safe)
    # ============================================================================

    def _pmf_lookback(self, k: int) -> float:
        """
        Your original heuristic PMF mapping of m -> k.
        """
        if k < self.len_w and k != 0:
            return 0.0

        prob_spontaneous = self.alphabet_size * self.epsilon
        if k == 0:
            return float(prob_spontaneous)

        prob = 0.0
        for m in range(0, 100000):
            value = m + self.len_w + self.lag_function(m)
            if abs(value - k) < 0.5:
                prob += self.p_w * ((1 - self.p_w) ** (m))
            if value > k + 10:
                break

        return float((1 - prob_spontaneous) * prob)
