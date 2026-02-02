import numpy as np
import random
from typing import List, Tuple, Set, Dict, Optional
# from itertools import product

class GarciaContextTreeSimulator:
    """
    Perfect simulation for locally continuous chains with unbounded memory.
    Implements Garcia (2011) construction with blocked rescaled coalescence times.
    
    Key differences from Gallo:
    - Uses blocked rescaling via θ^k sequence
    - Coalescence time Λ[0] instead of direct lookback L_n
    - Update function F with interval splitting of [0,1]
    """
    
    def __init__(
        self,
        alphabet: List[int],
        skeleton: Set[Tuple],
        alpha_sequence: Dict[Tuple, List[float]],
        epsilon: float = 0.1,
        max_blocks: int = 100,
        max_block_size: int = 50,
        show_progress: bool = False
    ):
        """
        Initialize Garcia simulator.
        
        Args:
            alphabet: State space A (e.g., [0, 1] or [-1, +1])
            skeleton: Probabilistic skeleton τ (set of contexts)
            alpha_sequence: For each context v, the sequence {α_k^v}_{k≥0}
                           where α_k^v = inf_{a_{-k}^{-1}} Σ_a inf_z P(a|v a_{-k}^{-1} z)
            epsilon: Non-nullness parameter (inf_z P(a|z) ≥ ε for all a)
            max_blocks: Maximum number of blocks to generate
            max_block_size: Maximum size of each block
            show_progress: Whether to show progress information
        """
        self.alphabet = alphabet
        self.skeleton = skeleton
        self.alpha_sequence = alpha_sequence
        self.epsilon = epsilon
        self.max_blocks = max_blocks
        self.max_block_size = max_block_size
        self.show_progress = show_progress
        
        # Derived quantities
        self.alphabet_size = len(self.alphabet)
        self.alpha_minus_1 = min(epsilon, 0.5)  # α_{-1} = min transition prob
        
        # Context length function
        self.max_context_length = max(len(ctx) for ctx in skeleton) if skeleton else 0
        
        if show_progress:
            print(f"[GarciaSim] Initialized with |A|={self.alphabet_size}, |τ|={len(self.skeleton)}")
            print(f"[GarciaSim] Max context length: {self.max_context_length}")
    
    # ============================================================================
    # CONTEXT TREE OPERATIONS
    # ============================================================================
    
    def find_context(self, past: List[int]) -> Tuple:
        """
        Find c_τ(past) = longest suffix of past that belongs to skeleton τ
        
        Returns:
            Context tuple (empty tuple if no context found)
        """
        if not past:
            return tuple()
        
        max_search = min(len(past), self.max_context_length * 2)
        
        for length in range(max_search, 0, -1):
            suffix = tuple(past[-length:])
            if suffix in self.skeleton:
                return suffix
        
        return tuple()  # Empty context
    
    def context_length_at_time(self, Y_sequence: List, time_idx: int) -> int:
        """
        Compute c_τ^n = sup_a |c_τ(Y_{-∞}^n(a))|
        
        Args:
            Y_sequence: Auxiliary process Y computed from U variables
            time_idx: Time index n
            
        Returns:
            Maximum context length at time n
        """
        if time_idx < 0:
            return 0
        
        # For each possible full context, compute context length
        # In practice, we only need to check the actual Y sequence
        past = Y_sequence[:time_idx + 1]
        context = self.find_context(past)
        return len(context)
    
    # ============================================================================
    # AUXILIARY PROCESS Y_n
    # ============================================================================
    
    def compute_Y_n(self, U_n: float, alpha_seq: List[float]) -> int:
        """
        Compute Y_n from U_n using interval splitting (Section 3.2):
        
        Y_n = Σ_{a ∈ A} a · 𝟙{Σ_{j=0}^{a-1} α(j) ≤ U_n < Σ_{j=0}^a α(j)}
              + ⋆ · 𝟙{U_n ≥ α(1)}
        
        where α(0) = 0, α(j) = inf_{z} P(j|z) for j ∈ A
        
        Args:
            U_n: Uniform random variable in [0,1]
            alpha_seq: Sequence of cumulative minima [α(0), α(1), ...]
            
        Returns:
            Symbol from alphabet or special marker -999 for ⋆
        """
        cumsum = 0.0
        
        # Check each symbol in alphabet
        for idx, symbol in enumerate(self.alphabet):
            alpha_current = alpha_seq[idx] if idx < len(alpha_seq) else 0
            
            if cumsum <= U_n < cumsum + alpha_current:
                return symbol
            
            cumsum += alpha_current
        
        # If U_n ≥ sum of all alphas, return ⋆ (marked as -999)
        return -999  # Special marker for ⋆
    
    def generate_Y_sequence(
        self, 
        U_sequence: List[float],
        past_for_context: Optional[List[int]] = None
    ) -> List[int]:
        """
        Generate auxiliary sequence Y from U sequence.
        
        Args:
            U_sequence: Sequence of uniform random variables
            past_for_context: Optional past to determine context-dependent alphas
            
        Returns:
            Y sequence (with -999 marking ⋆ positions)
        """
        Y_seq = []
        
        if past_for_context is None:
            past_for_context = []
        
        for U_n in U_sequence:
            # Find current context
            context = self.find_context(past_for_context)
            
            # Get alpha sequence for this context
            if context in self.alpha_sequence:
                alpha_seq = self.alpha_sequence[context]
            else:
                # Default: uniform distribution
                alpha_seq = [1.0 / self.alphabet_size] * self.alphabet_size
            
            # Compute Y_n
            Y_n = self.compute_Y_n(U_n, alpha_seq)
            Y_seq.append(Y_n)
            
            # Update past for next iteration
            if Y_n != -999:  # If not ⋆
                past_for_context.append(Y_n)
            # If ⋆, we need to use the actual symbol (but we don't have it yet in forward pass)
        
        return Y_seq
    
    # ============================================================================
    # GOOD COALESCENCE TIME (Definition 3.3)
    # ============================================================================
    
    def is_good_coalescence_time(
        self,
        Y_sequence: List[int],
        c_tau_sequence: List[int],
        candidate_time: int,
        window_start: int,
        window_end: int
    ) -> bool:
        """
        Check if time i is a good coalescence time for window [m,n].
        
        Time i is good if for all j ∈ [m,n]:
            Y_j ∈ A  OR  c_τ^j ≤ j - i
        
        Args:
            Y_sequence: Auxiliary process Y
            c_tau_sequence: Context lengths c_τ^j at each time j
            candidate_time: Time i to check
            window_start: m
            window_end: n
            
        Returns:
            True if candidate_time is a good coalescence time
        """
        for j in range(window_start, window_end + 1):
            Y_j = Y_sequence[j]
            c_tau_j = c_tau_sequence[j]
            
            # Check condition: Y_j ∈ A OR c_τ^j ≤ j - candidate_time
            in_alphabet = (Y_j != -999)  # Not ⋆
            context_bounded = (c_tau_j <= j - candidate_time)
            
            if not (in_alphabet or context_bounded):
                return False
        
        return True
    
    def compute_bar_theta(
        self,
        Y_sequence: List[int],
        c_tau_sequence: List[int],
        window_start: int,
        window_end: Optional[int] = None
    ) -> int:
        """
        Compute θ̄[m,n] = sup{i ≤ m : i is a good coalescence time for [m,n]}
        
        Args:
            Y_sequence: Auxiliary process
            c_tau_sequence: Context lengths at each time
            window_start: m
            window_end: n (defaults to m if None)
            
        Returns:
            Maximum good coalescence time (may be -∞ represented as large negative)
        """
        if window_end is None:
            window_end = window_start
        
        # Search backwards from window_start
        for i in range(window_start, -self.max_blocks * self.max_block_size - 1, -1):
            if self.is_good_coalescence_time(
                Y_sequence, c_tau_sequence, i, window_start, window_end
            ):
                return i
        
        # If no good coalescence time found, return very negative
        return -self.max_blocks * self.max_block_size
    
    # ============================================================================
    # BLOCK CONSTRUCTION (Section 3.4)
    # ============================================================================
    
    def construct_blocks(
        self,
        Y_sequence: List[int],
        c_tau_sequence: List[int],
        target_time: int = 0
    ) -> Tuple[List[int], List[Tuple[int, int]]]:
        """
        Construct blocks {B_k}_{k≥0} via θ^k recursion:
            θ^{-1} := 1
            θ^k = θ̄[θ^{k-1} - 1]
            B_k = {θ^k, ..., θ^{k-1} - 1}
        
        Args:
            Y_sequence: Auxiliary process
            c_tau_sequence: Context lengths
            target_time: Time point to reach (usually 0)
            
        Returns:
            (theta_sequence, blocks) where:
                theta_sequence = [θ^{-1}, θ^0, θ^1, ...]
                blocks = [(start, end) for each block]
        """
        theta_sequence = [target_time + 1]  # θ^{-1} = 1 (or target + 1)
        blocks = []
        
        for k in range(self.max_blocks):
            # Compute θ^k = θ̄[θ^{k-1} - 1]
            theta_prev = theta_sequence[-1]
            theta_k = self.compute_bar_theta(
                Y_sequence, c_tau_sequence, theta_prev - 1
            )
            
            theta_sequence.append(theta_k)
            
            # Define block B_k = {θ^k, ..., θ^{k-1} - 1}
            blocks.append((theta_k, theta_prev - 1))
            
            # Check if we've reached far enough into the past
            if theta_k < target_time - self.max_blocks * self.max_block_size:
                break
        
        return theta_sequence, blocks
    
    # ============================================================================
    # ζ_i and L_k COMPUTATION
    # ============================================================================
    
    def compute_zeta_i(
        self,
        U_i: float,
        c_tau_prev: int,
        alpha_seq: List[float]
    ) -> int:
        """
        Compute ζ_i as defined in Section 3.4:
        
        ζ_i = 𝟙{U_i ≥ α_{-1}} · Σ_{k≥0} k · 𝟙{U_i ∈ [α_{k-1}^{c_τ^{i-1}}, α_k^{c_τ^{i-1}})}
        
        Args:
            U_i: Uniform random variable at time i
            c_tau_prev: Context length c_τ^{i-1}
            alpha_seq: Alpha sequence for the context (α_k^v values)
            
        Returns:
            ζ_i value
        """
        # Check if U_i < α_{-1} (spontaneous generation)
        if U_i < self.alpha_minus_1:
            return 0
        
        # Search for k such that U_i ∈ [α_{k-1}, α_k)
        cumsum = self.alpha_minus_1
        
        for k in range(len(alpha_seq)):
            alpha_k = alpha_seq[min(k, len(alpha_seq) - 1)]
            
            if cumsum <= U_i < cumsum + alpha_k:
                return k
            
            cumsum += alpha_k
        
        # If beyond all intervals, return large value
        return len(alpha_seq)
    
    def compute_L_k(
        self,
        U_block: List[float],
        c_tau_block: List[int],
        block_indices: Tuple[int, int]
    ) -> int:
        """
        Compute L_k = sup_{i ∈ B_k} ζ_i
        
        Args:
            U_block: U values in block B_k
            c_tau_block: Context lengths in block
            block_indices: (start, end) indices of block
            
        Returns:
            L_k value
        """
        max_zeta = 0
        
        start_idx, end_idx = block_indices
        
        for i, (U_i, c_tau_i) in enumerate(zip(U_block, c_tau_block)):
            # Get context at time i
            # (In practice, we'd need to determine which context to use)
            alpha_seq = [self.epsilon] * self.alphabet_size  # Simplified
            
            zeta_i = self.compute_zeta_i(U_i, c_tau_i, alpha_seq)
            max_zeta = max(max_zeta, zeta_i)
        
        return max_zeta
    
    # ============================================================================
    # BLOCKED RESCALED COALESCENCE TIME Θ[k]
    # ============================================================================
    
    def compute_Theta(
        self,
        L_sequence: List[int],
        target_block: int = 0
    ) -> int:
        """
        Compute Θ[k] as in Eq (3.10):
        
        Θ[k] = sup{n ≤ k : L_i ≤ i - n, for i = n, ..., k}
        
        Args:
            L_sequence: Sequence of L_k values
            target_block: Block index k (usually 0)
            
        Returns:
            Θ[k] value
        """
        for n in range(target_block, -len(L_sequence) - 1, -1):
            # Check if L_i ≤ i - n for all i = n, ..., k
            valid = True
            
            for i in range(max(n, 0), target_block + 1):
                if i < len(L_sequence):
                    if L_sequence[i] > i - n:
                        valid = False
                        break
            
            if valid:
                return n
        
        return -len(L_sequence)
    
    def compute_Lambda(
        self,
        Theta_value: int,
        blocks: List[Tuple[int, int]]
    ) -> int:
        """
        Compute Λ[0] = -Σ_{i=0}^{-Θ[0]} |B_i|
        
        Args:
            Theta_value: Θ[0] value
            blocks: List of (start, end) for each block
            
        Returns:
            Λ[0] coalescence time
        """
        if Theta_value >= 0:
            return 0
        
        total_length = 0
        
        for i in range(-Theta_value + 1):
            if i < len(blocks):
                start, end = blocks[i]
                block_length = end - start + 1
                total_length += block_length
        
        return -total_length
    
    # ============================================================================
    # UPDATE FUNCTION F
    # ============================================================================
    
    def interval_length(
        self,
        symbol: int,
        past: List[int],
        depth: int
    ) -> float:
        """
        Compute |I_k(a|a_{-k}^{-1})| for interval splitting.
        
        |I_0(a|∅)| = α(a) = inf_z P(a|z)
        |I_k(a|a_{-k}^{-1})| = inf_z P(a|a_{-k}^{-1}z) - inf_z P(a|a_{-k+1}^{-1}z)
        
        Args:
            symbol: Target symbol a
            past: Past sequence
            depth: Depth k
            
        Returns:
            Interval length
        """
        if depth == 0:
            return self.epsilon  # α(a)
        
        # Simplified: assume uniform decay
        return self.epsilon / (depth + 1)
    
    def update_function_F(
        self,
        U_0: float,
        past: List[int]
    ) -> int:
        """
        Update function F(U_0, past) using interval splitting.
        
        F(U_0, a) = Σ_{a ∈ A} a · 𝟙{U_0 ∈ ⋃_k I_k(a|a_{-k}^{-1})}
        
        Args:
            U_0: Uniform random variable
            past: Historical sequence
            
        Returns:
            Next symbol
        """
        cumsum = 0.0
        
        for symbol in self.alphabet:
            # Sum intervals for this symbol
            for depth in range(min(len(past), self.max_context_length) + 1):
                interval_len = self.interval_length(symbol, past, depth)
                
                if cumsum <= U_0 < cumsum + interval_len:
                    return symbol
                
                cumsum += interval_len
        
        # Default: return first symbol
        return self.alphabet[0]
    
    # ============================================================================
    # PERFECT SIMULATION ALGORITHM
    # ============================================================================
    
    def perfect_simulation(
        self,
        target_time: int = 0,
        U_sequence: Optional[List[float]] = None
    ) -> Tuple[int, Dict]:
        """
        Algorithm 1: Perfect Simulation for Locally Continuous Chains
        
        Args:
            target_time: Target time n (usually 0)
            U_sequence: Pre-generated uniform random variables (optional)
            
        Returns:
            (X_n, info_dict) where:
                X_n: Sample from stationary distribution
                info_dict: Diagnostic information
        """
        # Step 1: Generate uniform random variables
        if U_sequence is None:
            seq_length = self.max_blocks * self.max_block_size
            U_sequence = [random.random() for _ in range(seq_length)]
        
        # Step 2-6: Generate Y sequence and context lengths
        Y_sequence = self.generate_Y_sequence(U_sequence)
        
        c_tau_sequence = []
        for i in range(len(Y_sequence)):
            c_tau_i = self.context_length_at_time(Y_sequence, i)
            c_tau_sequence.append(c_tau_i)
        
        # Step 7-12: Construct blocks
        theta_sequence, blocks = self.construct_blocks(
            Y_sequence, c_tau_sequence, target_time
        )
        
        # Compute L_k for each block
        L_sequence = []
        for k, (start, end) in enumerate(blocks):
            block_U = U_sequence[start:end+1]
            block_c_tau = c_tau_sequence[start:end+1]
            L_k = self.compute_L_k(block_U, block_c_tau, (start, end))
            L_sequence.append(L_k)
        
        # Step 15-16: Compute Θ[n] and Λ[n]
        Theta_n = self.compute_Theta(L_sequence, target_block=0)
        Lambda_n = self.compute_Lambda(Theta_n, blocks)
        
        # Step 19-25: Forward construction
        X_sequence = []
        for j in range(Lambda_n, target_time + 1):
            # Apply update function
            X_j = self.update_function_F(U_sequence[j], X_sequence)
            X_sequence.append(X_j)
        
        # Step 26: Return sample
        X_n = X_sequence[-1] if X_sequence else self.alphabet[0]
        
        info_dict = {
            'Lambda_n': Lambda_n,
            'Theta_n': Theta_n,
            'num_blocks': len(blocks),
            'L_sequence': L_sequence,
            'coalescence_time': -Lambda_n
        }
        
        return X_n, info_dict
    
    # ============================================================================
    # EXPECTATION BOUNDS
    # ============================================================================
    
    def expectation_bound_Lambda(self) -> Dict[str, float]:
        """
        Compute E[|Λ[0]|] using Theorem 3.5 conditions.
        
        Returns:
            Dictionary with bound information
        """
        # Compute E[θ̄[0]]
        E_bar_theta = 10.0  # Placeholder: should be computed from skeleton
        
        # Compute A_k sequence
        A_sequence = []
        for k in range(1, 100):
            # A_k = {1 - (E|θ̄[0]| + 1)P(U_0 > α_k^{c_τ^{-1}})} ∨ α_{-1}
            prob_exceed = np.exp(-k * 0.1)  # Simplified decay
            A_k = max(1 - (E_bar_theta + 1) * prob_exceed, self.alpha_minus_1)
            A_sequence.append(A_k)
        
        # Check convergence conditions
        # (i) Σ Π A_k = +∞
        prod_sum = sum(np.prod(A_sequence[:k]) for k in range(1, len(A_sequence)))
        
        # (ii) Σ (1 - A_k) < +∞
        sum_complement = sum(1 - A_k for A_k in A_sequence)
        
        return {
            'E_bar_theta': E_bar_theta,
            'prod_sum': prod_sum,
            'sum_complement': sum_complement,
            'feasible': prod_sum > 100 and sum_complement < np.inf,
            'regime': 'summable_tail' if sum_complement < np.inf else 'exponential_tail'
        }
    
    # ============================================================================
    # VALIDATION
    # ============================================================================
    
    def validate_perfect_simulation(
        self,
        num_samples: int = 1000
    ) -> Dict[str, float]:
        """
        Validate perfect simulation by generating samples and computing statistics.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Statistics dictionary
        """
        samples = []
        coalescence_times = []
        
        for _ in range(num_samples):
            X_n, info = self.perfect_simulation()
            samples.append(X_n)
            coalescence_times.append(info['coalescence_time'])
        
        return {
            'mean_coalescence_time': np.mean(coalescence_times),
            'std_coalescence_time': np.std(coalescence_times),
            'max_coalescence_time': np.max(coalescence_times),
            'sample_mean': np.mean(samples),
            'sample_std': np.std(samples),
            'num_samples': num_samples
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def create_simple_garcia_simulator():
    """
    Create a simple Garcia simulator with basic skeleton.
    """
    alphabet = [0, 1]
    
    # Simple skeleton: contexts of length 0, 1, 2
    skeleton = {
        tuple(),  # Empty context
        (0,), (1,),  # Length 1
        (0, 0), (0, 1), (1, 0), (1, 1)  # Length 2
    }
    
    # Alpha sequences for each context
    alpha_sequence = {}
    for ctx in skeleton:
        # α_k^v converges to 1 as k → ∞
        alpha_sequence[ctx] = [0.5 + 0.4 * (1 - np.exp(-k)) for k in range(10)]
    
    return GarciaContextTreeSimulator(
        alphabet=alphabet,
        skeleton=skeleton,
        alpha_sequence=alpha_sequence,
        epsilon=0.1,
        max_blocks=50,
        max_block_size=20,
        show_progress=True
    )