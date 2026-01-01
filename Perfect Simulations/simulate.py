from src.simulators.cff import BinaryAutoregressiveSimulator
import time
import matplotlib.pyplot as plt
theta0 = 0.2
class    InfiniteExponentialTheta:
    """
    Represents theta_k = alpha * rho^k for k >= 1.
    Behaves like a dictionary for finite keys and supports analytic tail sums.
    """
    def __init__(self, alpha=0.2, rho=0.7):
        assert 0 < rho < 1, "rho must be in (0,1) for exponential decay"
        self.alpha = alpha
        self.rho = rho

    def __getitem__(self, k):
        if k < 1:
            return 0.0
        return self.alpha * (self.rho ** k)

    def tail_sum(self, k):
        """
        r_k = sum_{m>k} |theta_m|
            = alpha * rho^(k+1) / (1 - rho)
        """
        return abs(self.alpha) * (self.rho ** (k + 1)) / (1 - self.rho)

    def max_k_effective(self, tol=1e-12):
        """
        Gives the largest k such that theta_k > tol.
        Useful for finite approximations if needed.
        """
        if self.alpha == 0: 
            return 0
        import math
        # solve alpha * rho^k = tol
        k = math.log(tol / abs(self.alpha)) / math.log(self.rho)
        return max(1, int(k))


class LongExponentialTheta:
    """
    A long but finite exponential-decay sequence:
        theta_k = alpha * rho^k
    with a hard guarantee that tail_sum(k) < 1 for all k.
    """
    def __init__(self, K_max, alpha=0.2, rho=0.7):
        assert 0 < rho < 1, "rho must be in (0,1)"
        # Enforce the tail < 1 condition
        # assert alpha < (1 - rho) / rho, \
        #     f"alpha must be < {(1-rho)/rho} to ensure tail_sum < 1"

        self.alpha = alpha
        self.rho = rho
        self.K_max = int(K_max)

    def __len__(self):
        return self.K_max

    def __getitem__(self, k):
        if 1 <= k <= self.K_max:
            return self.alpha * (self.rho ** k)
        return 0.0

    def tail_sum(self, k):
        """
        Finite geometric remainder:
            sum_{m=k+1}^{K_max} alpha rho^m
        Guaranteed < 1 by the parameter constraint.
        """
        if k >= self.K_max:
            return 0.0

        first = self.alpha * (self.rho ** (k + 1))
        terms = self.K_max - k
        return first * (1 - self.rho**terms) / (1 - self.rho)
    

class InfinitePolynomialTheta:
    """
    Represents theta_k = alpha / k^beta for k >= 1.
    Behaves like a dictionary for finite keys and supports analytic tail sums.
    """
    def __init__(self, alpha=0.2, beta=2.0):
        assert beta > 1, "beta must be > 1 for summability"
        self.alpha = alpha
        self.beta = beta

    def __getitem__(self, k):
        if k < 1:
            return 0.0
        return self.alpha / (k ** self.beta)

    def tail_sum(self, k):
        """
        r_k = sum_{m>k} |theta_m|
            <= alpha * integral_{k}^{infinity} x^{-beta} dx
            = alpha * (k^(1-beta)) / (beta - 1)
        """

        if k == 0:
            return 0.0
        else:
            return abs(self.alpha) * (k ** (1 - self.beta)) / (self.beta - 1)

    def max_k_effective(self, tol=1e-12):
        """
        Gives the largest k such that theta_k > tol.
        Useful for finite approximations if needed.
        """
        if self.alpha == 0: 
            return 0
        import math
        # solve alpha / k^beta = tol
        k = (abs(self.alpha) / tol) ** (1 / self.beta)
        return max(1, int(k))
    

window = (0, 1)


times_dct = {}
biases_dict = {}
max_regen_search_depth = 5000
for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    time_list, avg_user_biases = [], []
    for rho in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]:
        print(f"rho={rho:.1f}")
        
        theta_seq = InfiniteExponentialTheta(alpha=alpha, rho=rho)
        start_time = time.time()
        sim = BinaryAutoregressiveSimulator(theta0=0.1, theta_seq=theta_seq, max_regen_search_depth=max_regen_search_depth)
        perfect_sample, avg_user_bias = sim.perfect_sample(window=window)
        elapsed_time = time.time() - start_time

        time_list.append((rho, elapsed_time))
        avg_user_biases.append((rho, avg_user_bias))
    times_dct[alpha] = time_list
    biases_dict[alpha] = avg_user_biases

# Plotting
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(18, 10))
ax = ax.flatten()  # Flatten 2D array to 1D for easier indexing

for idx, (alpha, avg_user_biases) in enumerate(biases_dict.items()):
    rhos, biases = zip(*biases_dict[alpha])
    if idx < len(ax):  # Make sure we don't exceed the number of subplots
        ax[idx].plot(rhos, biases, '-o', label=f"alpha ={alpha}")
        ax[idx].set_xlabel("rho (upper bound of memory decay)")
        ax[idx].set_ylabel("Regeneration Time (s)")
        ax[idx].set_title(f"Regeneration Time vs rho (alpha={alpha})")
        ax[idx].legend()

plt.tight_layout()
plt.savefig("Regen_time_vs_rho:exp_decay.png")
plt.show()



times_dct = {}
biases_dict = {}
max_regen_search_depth = 5000
for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    time_list, avg_user_biases = [], []
    for beta in [2, 3,4,5,6,7,8,9]:
        print(f"beta={beta:.1f}")
        
        theta_seq = InfinitePolynomialTheta(alpha=alpha, beta=beta)
        start_time = time.time()
        sim = BinaryAutoregressiveSimulator(theta0=0.1, theta_seq=theta_seq, max_regen_search_depth=max_regen_search_depth)
        perfect_sample, avg_user_bias = sim.perfect_sample(window=window)
        elapsed_time = time.time() - start_time

        time_list.append((beta, elapsed_time))
        avg_user_biases.append((beta, avg_user_bias))
    times_dct[alpha] = time_list
    biases_dict[alpha] = avg_user_biases

# Plotting
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(18, 10))
ax = ax.flatten()  # Flatten 2D array to 1D for easier indexing

for idx, (alpha, avg_user_biases) in enumerate(biases_dict.items()):
    rhos, biases = zip(*biases_dict[alpha])
    if idx < len(ax):  # Make sure we don't exceed the number of subplots
        ax[idx].plot(rhos, biases, '-o', label=f"alpha ={alpha}")
        ax[idx].set_xlabel("rho (upper bound of memory decay)")
        ax[idx].set_ylabel("Regeneration Time (s)")
        ax[idx].set_title(f"Regeneration Time vs rho (alpha={alpha})")
        ax[idx].legend()

plt.tight_layout()
plt.savefig("Regen_time_vs_rho:poly_decay.png")
plt.show()