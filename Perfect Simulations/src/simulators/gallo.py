import numpy as np
import random
import math
from typing import List, Tuple, Set, Dict
from itertools import product

class GalloContextTreeSimulator:
    """
    Perfect simulation for chains with unbounded variable-length memory.
    Implements Gallo (2009) construction with context trees.
    
    Matches the experimental framework used in CFF experiments.
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
        show_progress: bool = False
    ):
        """
        Initialize Gallo simulator.
        
        Args:
            alpha: Growth parameter for lag function l_w(k) = exp(α·k)
            alphabet: State space (e.g., [-1, +1])
            reference_string: Reference string w (e.g., [-1, +1])
            epsilon: Minimum transition probability (non-nullness bound)
            beta: AR coefficient decay exponent: a_i = exp(-i^β)
            max_depth: Maximum context tree depth for generation (this is S)
            max_trie_depth: Maximum depth for context tree construction
            show_progress: Whether to show progress bars
        """
        self.alpha = alpha
        self.alphabet = alphabet
        self.reference_string = tuple(reference_string)
        self.epsilon = epsilon
        self.beta = beta
        self.max_depth = max_depth
        self.max_trie_depth = max_trie_depth
        self.show_progress = show_progress
        
        # Derived quantities
        self.len_w = len(self.reference_string)
        self.alphabet_size = len(self.alphabet)
        self.lag_function = lambda k: int(np.ceil(np.exp(alpha * k)))
        self.ar_coef = lambda i: np.exp(-i ** beta)
        self.p_w = epsilon ** self.len_w  # Probability of spontaneous w
        
        # Generate context tree τ as per Eq 9.5
        self.contexts = self._generate_contexts()
        
        if show_progress:
            print(f"[GalloSim] Initialized with α={alpha:.3f}, |τ|={len(self.contexts)} contexts")
    
    # ============================================================================
    # CONTEXT TREE GENERATION
    # ============================================================================
    
    def _generate_contexts(self) -> Set[Tuple]:
        """
        Generate τ = ⋃_{i≥0} ⋃_{c∈A^{l_w(i)}} c·w·A^i
        
        Each context has structure: filler + reference_string + prefix
        where |filler| = l_w(i) and |prefix| = i
        """
        contexts = set()
        w = self.reference_string
        
        for i in range(self.max_trie_depth + 1):
            lag_len = self.lag_function(i)
            
            # Generate all fillers of length l_w(i)
            for filler in product(self.alphabet, repeat=lag_len):
                # Generate all prefixes of length i
                for prefix in product(self.alphabet, repeat=i):
                    # Context = filler + w + prefix
                    context = tuple(filler) + w + tuple(prefix)
                    contexts.add(context)
        
        return contexts
    
    def find_context(self, past: List[int]) -> Tuple:
        """
        Find c_τ(past) = longest suffix of past that belongs to τ
        
        Returns:
            Context tuple (empty tuple if no context found)
        """
        max_search = min(len(past), self.max_trie_depth * 10)
        
        for length in range(max_search, 0, -1):
            suffix = tuple(past[-length:])
            if suffix in self.contexts:
                return suffix
        
        return tuple()  # Empty context (renewal point)
    
    # ============================================================================
    # LOOKBACK DEPTH COMPUTATION
    # ============================================================================
    
    def _find_last_w_distance(self, past: List[int]) -> float:
        """
        Find m_i = inf{k ≥ 0: past[-(k+|w|):-(k)] = w}
        
        Returns:
            Distance k to the last occurrence of w in past
            Returns np.inf if w not found
        """
        w = self.reference_string
        len_w = self.len_w
        
        for k in range(len(past) - len_w + 1):
            end_idx = len(past) - k
            start_idx = end_idx - len_w
            
            if start_idx >= 0 and tuple(past[start_idx:end_idx]) == w:
                return float(k)
        
        return np.inf
    
    def compute_lookback_depth(self, U_n: float, past: List[int]) -> float:
        """
        Compute L'_n as in Definition 6.1.4 (Eq 6.8):
        
        L'_n = { 0                        if U_n < #E·ε (spontaneous)
               { m_n + |w| + l_w(m_n)     otherwise
        
        Args:
            U_n: Uniform random variable in [0,1]
            past: Historical sequence (must have length ≥ 2S for safety)
        
        Returns:
            Lookback depth (can be np.inf if w not found)
        """
        # Spontaneous generation (independent of past)
        if U_n < self.alphabet_size * self.epsilon:
            return 0.0
        
        # Find distance to last occurrence of w
        m_n = self._find_last_w_distance(past)
        
        if m_n == np.inf:
            return np.inf
        
        # Compute required context length
        m_n_int = int(m_n)
        return float(m_n_int + self.len_w + self.lag_function(m_n_int))
    
    # ============================================================================
    # TRUNCATED EXPECTATION (Analytical)
    # ============================================================================

    def truncated_expectation_analytical(self) -> float:
        """
        Compute E[ψ_S(L_n)] = E[min(L_n, S)] analytically.
        
        This is EXACT μ_S from theory (no Monte Carlo error).
        
        Formula:
            μ_S = sum_{k=0}^{S-1} k·P(L_n = k) + S·P(L_n ≥ S)
        
        Returns:
            Exact truncated expectation
        """
        S = self.max_depth
        expect_sum = 0.0
        
        # Sum over k = 0, 1, ..., S-1
        for k in range(S):
            pk = self._pmf_lookback(k)
            expect_sum += k * pk
        
        # Add contribution from truncated tail: S·P(L_n ≥ S)
        prob_exceed_S = 1.0 - self._cdf_lookback(S)
        expect_sum += S * prob_exceed_S
        
        return expect_sum
    
    def _pmf_lookback(self, k: int) -> float:
        """
        Compute P(L_n = k) using the geometric-exponential model.
        
        P(L_n = k) = P(M + |w| + exp(α·M) = k) where M ~ Geom(p_w)
        """
        return pmf_M_plus_exp(k, self.p_w, self.len_w, self.alpha)
    
    def _cdf_lookback(self, k: int) -> float:
        """Compute P(L_n ≤ k)"""
        return sum(self._pmf_lookback(j) for j in range(k + 1))
    
    # ============================================================================
    # TAIL BOUNDS
    # ============================================================================
    
    def tail_expectation_exponential_gallo(self) -> float:
        """
        Compute E[(L_n - S)_+] for Gallo model.
        
        Uses Proposition 9.2.6: tail bound accounts for P(L_n > S) via:
            tail = r^{k_S-1} · (|w| - S + k_S - 1 + 1/p_w) 
                   + (p_w · r^{k_S-1} · e^{α·k_S}) / (1 - r·e^α)
        
        where k_S = min{k: |w| + k + exp(α·k) > S}
        
        Returns:
            Upper bound on tail expectation
        """
        S = self.max_depth
        r = 1 - self.p_w
        
        # Compute cutoff k_S
        k_S = 0
        while self.len_w + k_S + np.exp(self.alpha * k_S) <= S:
            k_S += 1
        
        if k_S <= 1:
            return 0.0
        
        r_power = r ** (k_S - 1)
        
        term1 = r_power * (self.len_w - S + (k_S - 1) + 1.0 / self.p_w)
        
        denominator = 1 - r * np.exp(self.alpha)
        if denominator <= 0:
            return np.inf
        
        term2 = (self.p_w * r_power * np.exp(self.alpha * k_S)) / denominator
        
        return max(0.0, term1 + term2)
    
    # ============================================================================
    # TRUNCATED ANALYTICAL BOUND
    # ============================================================================
    
    def truncated_analytical_bound(self) -> Dict[str, float]:
        """
        Proposition 9.2.6: E[L_n] ≤ μ_S + tail_bound
        
        Returns:
            Dictionary with bound components
        """
        # Compute μ_S analytically (EXACT)
        mu_S = self.truncated_expectation_analytical()
        
        if mu_S == np.inf:
            return {
                'mu_S_analytical': np.inf,
                'tail_bound': np.inf,
                'total_bound': np.inf,
                'truncation_index': self.max_depth,
                'method': 'truncated analytical (divergent)',
                'prob_exceed_S': 1.0
            }
        
        # Tail bound
        tail_bound = self.tail_expectation_exponential_gallo()
        
        # Scaling factor
        scaling = 1 - self.alphabet_size * self.epsilon
        
        # Probability of exceeding S
        prob_exceed_S = 1.0 - self._cdf_lookback(self.max_depth)
        
        return {
            'mu_S_analytical': mu_S,
            'tail_bound': tail_bound,
            'total_bound': scaling * (mu_S + tail_bound),
            'truncation_index': self.max_depth,
            'method': 'truncated analytical (no MC error)',
            'prob_exceed_S': prob_exceed_S
        }
    
    # ============================================================================
    # NON-TRUNCATED ANALYTICAL BOUND
    # ============================================================================
    
    def non_truncated_expectation_analytical(self) -> float:
        """
        Compute exact E[L_n] for non-truncated perfect simulation.
        
        Uses summation E[L_n] = Σ_{k≥0} P(L_n > k)
        
        Returns:
            Exact expectation
        """
        scaling = 1 - self.alphabet_size * self.epsilon
        expectation = 0.0
        tolerance = 1e-12
        max_terms = 10000
        
        for k in range(max_terms):
            prob_exceed_k = 1.0 - self._cdf_lookback(k)
            expectation += prob_exceed_k
            
            if prob_exceed_k < tolerance:
                break
        
        return scaling * expectation
    
    def non_truncated_analytical_bound(self, compute_exact: bool = False) -> Dict[str, float]:
        """
        Theorem 6.1.16: E[L_n] ≤ (1 - #E·ε) * (1/p_w + |w| + p_w·M·κ)
        
        where:
            - p_w = ε^|w| (prob of spontaneous w)
            - κ = Σ_{k≥0} (e^α · (1 - p_w))^k
            - M = 1 for l_w(k) = exp(α·k)
        
        Returns:
            Dictionary with theoretical bound and optionally exact value
        """
        r = 1 - self.p_w
        exp_alpha = np.exp(self.alpha)
        
        # Check convergence of κ series
        if r * exp_alpha >= 1:
            theoretical_bound = np.inf
        else:
            kappa = 1.0 / (1 - r * exp_alpha)
            
            # Bound components
            term1 = (1 - self.p_w) / self.p_w
            term2 = self.len_w
            term3 = self.p_w * kappa  # M = 1
            
            scaling = 1 - self.alphabet_size * self.epsilon
            theoretical_bound = scaling * (term1 + term2 + term3)
        
        result = {
            'theoretical_bound': theoretical_bound,
            'decay_type': 'exponential_gallo',
            'method': 'non-truncated analytical bound',
            'source': 'Theorem 6.1.16'
        }
        
        # Optionally compute exact value
        if compute_exact:
            exact_value = self.non_truncated_expectation_analytical()
            result['exact_value'] = exact_value
            if theoretical_bound is not None and theoretical_bound < np.inf:
                result['gap'] = theoretical_bound - exact_value
                result['tightness'] = exact_value / theoretical_bound if theoretical_bound > 0 else 0
        
        return result
    
    # ============================================================================
    # UNIFIED INTERFACE (matching CFF API)
    # ============================================================================
    
    def analytical_lookback_bound(
        self,
        truncated: bool = True,
        compute_exact: bool = False,
        **kwargs
    ) -> Dict[str, float]:
        """
        Unified interface for analytical lookback bounds.
        
        Args:
            truncated: If True, use truncated bound (user-imposed limit S)
                      If False, use non-truncated bound (true perfect simulation)
            compute_exact: If True, compute exact E[L_n] by summation
            **kwargs: Additional arguments (for API compatibility)
            
        Returns:
            Dictionary with bound information
        """
        if truncated:
            return self.truncated_analytical_bound()
        else:
            return self.non_truncated_analytical_bound(compute_exact=compute_exact)
    
    # ============================================================================
    # EMPIRICAL SAMPLING AND VALIDATION
    # ============================================================================
    
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
        lookbacks = []
        
        for _ in range(num_samples):
            # Generate random past of length sufficient for lookback
            past_length = max(self.max_depth * 2, 1000)
            past = [random.choice(self.alphabet) for _ in range(past_length)]
            U_n = random.random()
            
            L_n = self.compute_lookback_depth(U_n, past)
            
            if truncated:
                lookbacks.append(min(L_n, self.max_depth))
            else:
                if L_n < np.inf:
                    lookbacks.append(L_n)
        
        return np.array(lookbacks)
    
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
        
        if len(samples) == 0:
            return {
                **analytical,
                'empirical_mean': np.nan,
                'empirical_std': np.nan,
                'empirical_std_error': np.nan,
                'discrepancy': np.nan,
                'relative_discrepancy': np.nan,
                'num_samples': 0,
                'truncated': truncated,
                'note': 'No finite samples generated'
            }
        
        # Statistics
        empirical_mean = np.mean(samples)
        empirical_std = np.std(samples)
        empirical_std_error = empirical_std / np.sqrt(len(samples))
        
        # Comparison
        if truncated:
            analytical_value = analytical['mu_S_analytical']
        else:
            analytical_value = analytical.get('exact_value', analytical['theoretical_bound'])
        
        discrepancy = analytical_value - empirical_mean
        relative_discrepancy = discrepancy / analytical_value if analytical_value > 0 else 0
        
        return {
            **analytical,
            'empirical_mean': empirical_mean,
            'empirical_std': empirical_std,
            'empirical_std_error': empirical_std_error,
            'discrepancy': discrepancy,
            'relative_discrepancy': relative_discrepancy,
            'num_samples': len(samples),
            'truncated': truncated
        }
    
    # ============================================================================
    # USER IMPATIENCE BIAS
    # ============================================================================
    
    def compute_user_impatience_bias_given_limit(self) -> float:
        """
        Compute user impatience bias as per Eq 9.1:
        
            bias = P(L_n > S) / P(L_n ≤ S)
        
        Returns:
            Bias ratio (np.inf if P(L_n > S) ≥ 0.9999)
        """
        prob_exceed = 1.0 - self._cdf_lookback(self.max_depth)
        
        if prob_exceed >= 0.9999:
            return np.inf
        
        return prob_exceed / (1 - prob_exceed)
    
    # ============================================================================
    # TRANSITION PROBABILITIES (AR Model from Eq 9.6-9.7)
    # ============================================================================
    
    def transition_probability(self, symbol: int, context: Tuple) -> float:
        """
        Compute p(symbol | context) via AR model (Eq 9.7):
        
        p(X_t = s | X_{t-1}, ..., X_{t-K}) = 
            exp(Σ_i a_i φ(X_{t-i}, s)) / Z
        
        where φ(x, s) = 𝟙{x = s} and a_i = exp(-i^β)
        """
        K = len(context)
        
        if K == 0:  # Empty context → uniform
            return 1.0 / self.alphabet_size
        
        # Compute score for target symbol
        score_s = sum(
            self.ar_coef(i + 1) 
            for i in range(K) 
            if context[-(i + 1)] == symbol
        )
        
        # Compute partition function Z
        Z = sum(
            np.exp(sum(
                self.ar_coef(i + 1) 
                for i in range(K) 
                if context[-(i + 1)] == s
            ))
            for s in self.alphabet
        )
        
        if Z == 0:
            return 1.0 / self.alphabet_size
        
        return np.exp(score_s) / Z


# ============================================================================
# HELPER FUNCTION: PMF Computation
# ============================================================================

def pmf_M_plus_exp(k: int, p_w: float, w: int, alpha: float, 
                   m_max: int = 10000, tol: float = 1e-12) -> float:
    """
    Computes P(M + w + exp(alpha·M) = k), where M ~ Geom(p_w)
    
    Parameters:
        k: Target value
        p_w: Geometric success probability (0 < p_w <= 1)
        w: Constant shift (|w| in the model)
        alpha: Exponent coefficient
        m_max: Upper bound for searching m
        tol: Numerical tolerance for equality check
    
    Returns:
        Probability
    """
    prob = 0.0
    
    for m in range(1, m_max + 1):
        value = m + w + math.exp(alpha * m)
        
        if abs(value - k) < tol:
            prob += p_w * (1 - p_w) ** (m - 1)
        
        # Early stopping if exp(alpha·m) explodes
        if alpha > 0 and value > k + 1:
            break
    
    return prob
