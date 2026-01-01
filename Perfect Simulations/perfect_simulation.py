import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

class PerfectSampler:
    """
    Class for performing perfect sampling from a stationary distribution
    of a finite Markov chain using regeneration (without coalescence).
    """

    def __init__(self, transition_matrix: np.ndarray, seed=None):
        """
        Initialize the sampler.

        Parameters:
        - transition_matrix: np.ndarray, shape (n_states, n_states)
          The transition matrix of the Markov chain.
        - seed: int or None
          Random seed for reproducibility.
        """
        self.P = transition_matrix
        self.n_states = self.P.shape[0]
        if seed is not None:
            np.random.seed(seed)
        self.beta = self._compute_beta()
        self.nu = self._compute_minorizing_measure()

    def _compute_beta(self) -> float:
        """
        Compute the ergodicity coefficient β(P).

        Returns:
        - beta: float
        """
        beta = np.inf
        for i in range(self.n_states):
            for j in range(self.n_states):
                beta = min(beta, np.sum(np.minimum(self.P[i], self.P[j])))
        return beta

    def _compute_minorizing_measure(self) -> np.ndarray:
        """
        Compute the minorizing measure ν.

        Returns:
        - nu: np.ndarray, shape (n_states,)
        """
        i, j = 0, 1  # Arbitrary two different rows
        min_vec = np.minimum(self.P[i], self.P[j])
        return min_vec / self.beta

    def _transition(self, x: int, u: float) -> int:
        """
        Apply the transition function F(x, u).

        Parameters:
        - x: int, current state index
        - u: float, uniform random number

        Returns:
        - int, next state index
        """
        return int(u >= self.P[x][0])

    def sample_one(self) -> int:
        """
        Perform one perfect sample and return the state at time 0.

        Returns:
        - x0: int, sampled state index at time 0
        """
        U = {}
        t = -1
        while True:
            U[t] = np.random.uniform()
            if U[t] < self.beta:
                T0 = t
                break
            t -= 1

        X = np.random.choice(range(self.n_states), p=self.nu)

        for s in range(T0, 0):
            X = self._transition(X, U[s])
        return X

    def sample_trajectory(self) -> list:
        """
        Perform one perfect sample and return the full trajectory.

        Returns:
        - trajectory: list of (t, state_index)
        """
        U = {}
        t = -1
        while True:
            U[t] = np.random.uniform()
            if U[t] < self.beta:
                T0 = t
                break
            t -= 1

        X = np.random.choice(range(self.n_states), p=self.nu)
        trajectory = [(T0, X)]

        for s in range(T0, 0):
            X = self._transition(X, U[s])
            trajectory.append((s + 1, X))

        return trajectory

    def run_simulation(self, n_samples: int) -> list:
        """
        Run multiple perfect simulations.

        Parameters:
        - n_samples: int, number of samples to generate

        Returns:
        - samples: list of sampled state indices at time 0
        """
        print(f"Running perfect sampling with {n_samples} samples...")
        return [self.sample_one() for _ in range(n_samples)]

    def plot_histogram(self, samples: list):
        """
        Plot histogram of sampled X₀ values.

        Parameters:
        - samples: list of state indices
        """
        labels = [x + 1 for x in samples]  # Convert to 1-based
        counts = Counter(labels)
        most_common, freq = counts.most_common(1)[0]

        print(f"Most frequent state at time 0: state {most_common} (appeared {freq} times)")

        plt.figure(figsize=(6, 4))
        plt.hist(labels, bins=np.arange(0.5, self.n_states + 1.5), rwidth=0.6, edgecolor='black')
        plt.xticks(range(1, self.n_states + 1))
        plt.xlabel("State")
        plt.ylabel("Frequency")
        plt.title("Histogram of Perfectly Sampled $X_0$")
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    def plot_trajectories(self, num_trajectories=5):
        """
        Plot several sample trajectories from T(0) to time 0.

        Parameters:
        - num_trajectories: int, number of sample paths to plot
        """
        plt.figure(figsize=(10, 6))
        for _ in range(num_trajectories):
            traj = self.sample_trajectory()
            times, states = zip(*traj)
            states = [s + 1 for s in states]  # Convert to 1-based
            plt.plot(times, states, marker='o')
            plt.plot(times[0], states[0], 's', color='black', markersize=8, label='T(0)' if _ == 0 else "")
            plt.plot(times[-1], states[-1], marker='x', color='black', markersize=8, label='0' if _ == 0 else "")
        plt.xticks(range(0, max(times) + 1))
        plt.yticks(range(1, self.n_states + 1))
        plt.axhline(y=1, linestyle='--', color='gray', alpha=0.4)
        if self.n_states > 1:
            plt.axhline(y=2, linestyle='--', color='gray', alpha=0.4)
        plt.xlabel("Time (t)")
        plt.ylabel("State")
        plt.title(f"Sample Trajectories from $T(0)$ to 0 for {num_trajectories} Samples")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

# Usage example
if __name__ == "__main__":
    # Example: two-state Markov chain
    P = np.array([
        [0.4, 0.6],
        [0.2, 0.8]
    ])

    sampler = PerfectSampler(P, seed=42)

    # Run and plot results
    samples = sampler.run_simulation(n_samples=100000)
    sampler.plot_histogram(samples)
    sampler.plot_trajectories(num_trajectories=10)
