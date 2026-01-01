import numpy as np
from typing import Callable, Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class Kernel:
    """Represents a transition probability kernel P(a|past)."""
    alphabet: List
    transition_prob: Callable[[any, tuple], float]
    
    def __call__(self, symbol, past):
        """P(symbol|past)"""
        return self.transition_prob(symbol, past)


class CFTPSimulator:
    """
    Coupling From The Past simulator using length function and θ'[0].
    
    Based on Gallo & Garcia (2012) Algorithm for locally continuous chains.
    """
    
    def __init__(self, kernel: Kernel, max_iterations: int = 10000):
        self.kernel = kernel
        self.alphabet = kernel.alphabet
        self.max_iterations = max_iterations
        
        # Precompute alpha values
        self.alpha = self._compute_alpha()
        self.alpha_sum = sum(self.alpha.values())
        
        # Storage
        self.U = {}  # Uniform random variables
        self.X = {}  # Constructed chain
        
    def _compute_alpha(self) -> Dict:
        """
        Compute α(a) = inf_z P(a|z) for each symbol a.
        
        For practical implementation, we approximate the infimum
        by sampling many random pasts.
        """
        alpha = {}
        num_samples = 1000
        
        for a in self.alphabet:
            min_prob = float('inf')
            
            # Sample random pasts to approximate infimum
            for _ in range(num_samples):
                # Generate random past of length 20
                past = tuple(np.random.choice(self.alphabet, size=20))
                prob = self.kernel(a, past)
                min_prob = min(min_prob, prob)
            
            alpha[a] = max(min_prob, 1e-10)  # Avoid zero
            
        return alpha
    
    def sample_from_alpha(self, u: float) -> any:
        """
        Sample symbol from level-0 intervals using α values.
        
        When u < α_sum, we can determine the symbol without any past.
        """
        cumulative = 0.0
        for symbol in self.alphabet:
            cumulative += self.alpha[symbol]
            if u < cumulative:
                return symbol
        return self.alphabet[-1]  # Shouldn't reach here
    
    def compute_omega_k(self, past_k: tuple, num_samples: int = 100) -> float:
        """
        Compute ω_k(past_k) = Σ_a inf_z P(a|past_k, z).
        
        This represents the cumulative probability up to level k.
        """
        omega = 0.0
        
        for a in self.alphabet:
            # Approximate inf_z P(a|past_k, z)
            min_prob = float('inf')
            
            for _ in range(num_samples):
                # Random continuation
                continuation = tuple(np.random.choice(self.alphabet, size=10))
                full_past = past_k + continuation
                prob = self.kernel(a, full_past)
                min_prob = min(min_prob, prob)
            
            omega += max(min_prob, 1e-10)
        
        return omega
    
    def compute_length(self, u: float, time: int, 
                       max_length: int = 50) -> int:
        """
        Compute length function L(u, constructed_past).
        
        Returns the minimum k such that u < ω_k(past_k).
        """
        # Level 0: no past needed
        if u < self.alpha_sum:
            return 0
        
        # Levels k = 1, 2, 3, ...
        for k in range(1, max_length + 1):
            # Check if we have enough constructed past
            if time - k not in self.X:
                return float('inf')  # Not enough past yet
            
            # Get past of length k
            past_k = tuple(self.X[time - k + i] for i in range(k))
            
            # Compute ω_k
            omega_k = self.compute_omega_k(past_k)
            
            if u < omega_k:
                return k
        
        return float('inf')  # Discontinuous or need more
    
    def check_coalescence_condition(self, start_time: int, 
                                    end_time: int) -> bool:
        """
        Check if θ'[start_time, end_time] = start_time.
        
        Condition: L(U_j, ...) ≤ j - start_time for all j ∈ [start_time, end_time]
        """
        for j in range(start_time, end_time + 1):
            if j not in self.U:
                return False
            
            # Compute length for this time
            L_j = self.compute_length(self.U[j], j, max_length=j - start_time)
            
            # Check condition
            if L_j > j - start_time:
                return False
        
        return True
    
    def sample_with_partition(self, u: float, past_k: tuple, k: int) -> any:
        """
        Sample symbol using the partition at level k.
        
        Given that L = k, determine which symbol u falls into.
        """
        # Build partition up to level k
        cumulative = 0.0
        
        for level in range(k + 1):
            if level == 0:
                # Level 0 intervals
                for a in self.alphabet:
                    width = self.alpha[a]
                    if cumulative <= u < cumulative + width:
                        return a
                    cumulative += width
            else:
                # Level 'level' intervals
                past_level = past_k[-level:] if level <= len(past_k) else past_k
                
                for a in self.alphabet:
                    # Compute interval width for I_level(a|past_level)
                    if level == 1:
                        width = (self._min_prob_given_past(a, past_level) - 
                                self.alpha[a])
                    else:
                        width = (self._min_prob_given_past(a, past_level) - 
                                self._min_prob_given_past(a, past_level[1:]))
                    
                    width = max(width, 0)  # Ensure non-negative
                    
                    if cumulative <= u < cumulative + width:
                        return a
                    cumulative += width
        
        # Default to most probable symbol
        probs = {a: self.kernel(a, past_k) for a in self.alphabet}
        return max(probs, key=probs.get)
    
    def _min_prob_given_past(self, symbol: any, past: tuple, 
                            num_samples: int = 50) -> float:
        """Approximate inf_z P(symbol|past, z)."""
        min_prob = float('inf')
        
        for _ in range(num_samples):
            continuation = tuple(np.random.choice(self.alphabet, size=10))
            full_past = past + continuation
            prob = self.kernel(symbol, full_past)
            min_prob = min(min_prob, prob)
        
        return max(min_prob, 1e-10)
    
    def simulate_single_site(self, time: int = 0, verbose: bool = False) -> any:
        """
        Main CFTP algorithm to simulate X[time] using θ'[time].
        
        Returns:
            Symbol from the stationary distribution at time 'time'
        """
        # Reset storage
        self.U = {}
        self.X = {}
        
        target_time = time
        current_time = time
        
        if verbose:
            print(f"Target time: {target_time}")
            print(f"α_sum = {self.alpha_sum:.4f}")
            print("\n" + "="*60)
        
        # Phase 1: Generate backwards until coalescence
        for iteration in range(self.max_iterations):
            # Generate uniform
            u = np.random.uniform(0, 1)
            self.U[current_time] = u
            
            if verbose and iteration % 10 == 0:
                print(f"\nIteration {iteration}: time = {current_time}")
            
            # Check for "free" sample (L = 0)
            if u < self.alpha_sum:
                # Jackpot! Can start here
                symbol = self.sample_from_alpha(u)
                self.X[current_time] = symbol
                
                if verbose:
                    print(f"  ✓ Found free sample: U = {u:.4f} < {self.alpha_sum:.4f}")
                    print(f"  ✓ X[{current_time}] = {symbol}")
                    print(f"  ✓ Coalescence time τ = {current_time}")
                
                tau = current_time
                break
            
            # Check coalescence condition from current_time
            if self.check_coalescence_condition(current_time, target_time):
                if verbose:
                    print(f"  ✓ Coalescence condition satisfied at time {current_time}")
                
                tau = current_time
                # Need to determine X[tau] from any starting past
                # Since condition holds, all pasts will give same result
                # Use arbitrary past
                arbitrary_past = tuple(np.random.choice(self.alphabet, size=50))
                L_tau = self.compute_length(self.U[tau], tau, max_length=50)
                self.X[tau] = self.sample_with_partition(
                    self.U[tau], 
                    arbitrary_past[-L_tau:] if L_tau < float('inf') else arbitrary_past,
                    min(L_tau, 50)
                )
                
                if verbose:
                    print(f"  ✓ X[{tau}] = {self.X[tau]} (L = {L_tau})")
                
                break
            
            # Continue backwards
            current_time -= 1
            
        else:
            raise RuntimeError(
                f"Failed to find coalescence time after {self.max_iterations} iterations"
            )
        
        if verbose:
            print("\n" + "="*60)
            print(f"\nPhase 2: Constructing forward from τ = {tau} to {target_time}")
            print("="*60)
        
        # Phase 2: Construct forward from tau to target_time
        for j in range(tau + 1, target_time + 1):
            # Compute length
            L_j = self.compute_length(self.U[j], j)
            
            if L_j == 0:
                # Direct sample
                self.X[j] = self.sample_from_alpha(self.U[j])
                if verbose:
                    print(f"Time {j}: L = 0, X[{j}] = {self.X[j]} (direct)")
            
            elif L_j < float('inf'):
                # Need past of length L_j
                past_L = tuple(self.X[j - L_j + i] for i in range(L_j))
                self.X[j] = self.sample_with_partition(self.U[j], past_L, L_j)
                
                if verbose:
                    print(f"Time {j}: L = {L_j}, past = {past_L}, X[{j}] = {self.X[j]}")
            
            else:
                raise RuntimeError(f"Infinite length at time {j}")
        
        if verbose:
            print("\n" + "="*60)
            print(f"✓ Successfully generated X[{target_time}] = {self.X[target_time]}")
            print(f"✓ Total backward steps: {target_time - tau}")
            print("="*60 + "\n")
        
        return self.X[target_time]
    
    def simulate_window(self, start: int, end: int, 
                       verbose: bool = False) -> Dict[int, any]:
        """
        Simulate the chain in window [start, end] using θ'[start, end].
        
        Returns:
            Dictionary mapping time -> symbol for times in [start, end]
        """
        # Reset
        self.U = {}
        self.X = {}
        
        current_time = start
        
        # Phase 1: Generate backwards
        for iteration in range(self.max_iterations):
            u = np.random.uniform(0, 1)
            self.U[current_time] = u
            
            # Check free sample
            if u < self.alpha_sum:
                self.X[current_time] = self.sample_from_alpha(u)
                tau = current_time
                break
            
            # Check coalescence for entire window
            if self.check_coalescence_condition(current_time, end):
                tau = current_time
                # Start with arbitrary past
                L_tau = self.compute_length(self.U[tau], tau, max_length=50)
                arbitrary_past = tuple(np.random.choice(self.alphabet, size=50))
                self.X[tau] = self.sample_with_partition(
                    self.U[tau],
                    arbitrary_past[-L_tau:] if L_tau < float('inf') else arbitrary_past,
                    min(L_tau, 50)
                )
                break
            
            current_time -= 1
        else:
            raise RuntimeError("Failed to find coalescence")
        
        # Phase 2: Construct forward
        for j in range(tau + 1, end + 1):
            L_j = self.compute_length(self.U[j], j)
            
            if L_j == 0:
                self.X[j] = self.sample_from_alpha(self.U[j])
            elif L_j < float('inf'):
                past_L = tuple(self.X[j - L_j + i] for i in range(L_j))
                self.X[j] = self.sample_with_partition(self.U[j], past_L, L_j)
            else:
                raise RuntimeError(f"Infinite length at {j}")
        
        # Return window
        return {t: self.X[t] for t in range(start, end + 1)}


# ============================================================================
# Example 1: Binary Autoregressive Process
# ============================================================================

class BinaryARKernel(Kernel):
    """
    Binary autoregressive process: P(1|past) = ψ(θ_0 + Σ θ_k * past[k])
    
    Example from Comets et al. (2002) and Gallo & Garcia (2012).
    """
    
    def __init__(self, theta_0: float, theta_weights: List[float], 
                 psi: Callable[[float], float] = None):
        self.theta_0 = theta_0
        self.theta = np.array(theta_weights)
        
        # Default: logistic function
        if psi is None:
            self.psi = lambda x: 1 / (1 + np.exp(-x))
        else:
            self.psi = psi
        
        super().__init__(
            alphabet=[-1, +1],
            transition_prob=self._compute_prob
        )
    
    def _compute_prob(self, symbol: int, past: tuple) -> float:
        """Compute P(symbol|past)."""
        # Compute weighted sum
        past_array = np.array(past[-len(self.theta):])
        weights = self.theta[:len(past_array)]
        
        linear_comb = self.theta_0 + np.dot(weights, past_array)
        
        # Probability of +1
        p_plus = self.psi(linear_comb)
        
        if symbol == 1:
            return p_plus
        else:
            return 1 - p_plus


# ============================================================================
# Example 2: Simple Markov Chain
# ============================================================================

class SimpleMarkovKernel(Kernel):
    """Simple 2-state Markov chain for testing."""
    
    def __init__(self, p_01: float = 0.3, p_10: float = 0.4):
        """
        Args:
            p_01: P(1|0) - probability of transitioning 0 -> 1
            p_10: P(0|1) - probability of transitioning 1 -> 0
        """
        self.p_01 = p_01
        self.p_10 = p_10
        
        super().__init__(
            alphabet=[0, 1],
            transition_prob=self._compute_prob
        )
    
    def _compute_prob(self, symbol: int, past: tuple) -> float:
        """Compute P(symbol|past) - only depends on past[-1]."""
        if len(past) == 0:
            # Uniform if no past
            return 0.5
        
        last_symbol = past[-1]
        
        if last_symbol == 0:
            return self.p_01 if symbol == 1 else (1 - self.p_01)
        else:
            return self.p_10 if symbol == 0 else (1 - self.p_10)


# ============================================================================
# Testing and Demonstrations
# ============================================================================

def test_simple_markov():
    """Test with simple Markov chain."""
    print("="*70)
    print("TEST 1: Simple Markov Chain")
    print("="*70)
    
    kernel = SimpleMarkovKernel(p_01=0.3, p_10=0.4)
    simulator = CFTPSimulator(kernel, max_iterations=100)
    
    # Single site
    print("\nGenerating single sample at time 0:")
    sample = simulator.simulate_single_site(time=0, verbose=True)
    
    # Multiple samples to check distribution
    print("\nGenerating 1000 samples to verify distribution:")
    samples = [simulator.simulate_single_site(time=0) for _ in range(1000)]
    
    empirical_p1 = sum(1 for s in samples if s == 1) / len(samples)
    
    # Theoretical stationary distribution
    p_10, p_01 = 0.4, 0.3
    pi_1 = p_01 / (p_01 + p_10)
    
    print(f"Empirical P(X=1): {empirical_p1:.3f}")
    print(f"Theoretical P(X=1): {pi_1:.3f}")
    print(f"Difference: {abs(empirical_p1 - pi_1):.3f}")


def test_binary_ar():
    """Test with binary autoregressive process."""
    print("\n" + "="*70)
    print("TEST 2: Binary Autoregressive Process")
    print("="*70)
    
    # Exponentially decaying weights
    theta_0 = 0.0
    theta_weights = [0.5 * (0.7**k) for k in range(10)]
    
    print(f"\nParameters:")
    print(f"  θ_0 = {theta_0}")
    print(f"  θ_k = 0.5 * 0.7^k, k=1,...,10")
    print(f"  ψ(x) = logistic(x)")
    
    kernel = BinaryARKernel(theta_0, theta_weights)
    simulator = CFTPSimulator(kernel, max_iterations=500)
    
    # Single sample
    print("\nGenerating single sample:")
    sample = simulator.simulate_single_site(time=0, verbose=True)
    
    # Window
    print("\nGenerating window [0, 10]:")
    window = simulator.simulate_window(0, 10, verbose=False)
    print(f"Generated: {[window[t] for t in range(11)]}")


def test_convergence_rates():
    """Test convergence rate for different parameter settings."""
    print("\n" + "="*70)
    print("TEST 3: Convergence Rate Analysis")
    print("="*70)
    
    decay_rates = [0.5, 0.7, 0.9]
    
    for decay in decay_rates:
        print(f"\n--- Decay rate: {decay} ---")
        
        theta_weights = [0.5 * (decay**k) for k in range(15)]
        kernel = BinaryARKernel(0.0, theta_weights)
        simulator = CFTPSimulator(kernel, max_iterations=1000)
        
        backward_steps = []
        for _ in range(20):
            simulator.simulate_single_site(time=0, verbose=False)
            # Count backward steps (time 0 minus tau)
            tau = min(simulator.X.keys())
            steps = 0 - tau
            backward_steps.append(steps)
        
        print(f"Average backward steps: {np.mean(backward_steps):.1f}")
        print(f"Max backward steps: {np.max(backward_steps)}")
        print(f"Min backward steps: {np.min(backward_steps)}")


if __name__ == "__main__":
    # Run tests
    test_simple_markov()
    test_binary_ar()
    test_convergence_rates()