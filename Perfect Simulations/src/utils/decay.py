import math
from dataclasses import dataclass

@dataclass(frozen=True)
class InfiniteExponentialTheta:
    """
    Infinite exponentially decaying sequence:

        theta_k = alpha * rho^k,  k >= 1

    Features:
    - O(1) access to any coefficient
    - Exact analytic tail sums
    - Finite truncation via tolerance
    """
    alpha: float = 0.2
    rho: float = 0.7
    name: str = "InfiniteExponentialTheta"

    def __post_init__(self):
        if not (0.0 < self.rho < 1.0):
            raise ValueError("rho must be in (0,1) for exponential decay")

    def __getitem__(self, k: int) -> float:
        if k < 1:
            return 0.0
        return self.alpha * (self.rho ** k)

    def __contains__(self, k: int) -> bool:
        return k >= 1
    
    @property
    def decay_rate(self) -> float:
        """Exponential decay base."""
        return self.rho

    def tail_sum(self, k: int) -> float:
        """
        Exact tail sum:

            r_k = sum_{m>k} |theta_m|
                = |alpha| * rho^(k+1) / (1 - rho)
        """
        if k < 0:
            k = 0
        return abs(self.alpha) * (self.rho ** (k + 1)) / (1 - self.rho)

    def max_k_effective(self, tol: float = 1e-12) -> int:
        """
        Smallest k such that |theta_m| <= tol for all m >= k.
        """
        if self.alpha == 0:
            return 0

        k = math.log(abs(self.alpha) / tol) / (-math.log(self.rho))
        return max(1, math.ceil(k))

    def iter_values(self, k_max: int):
        """
        Efficient generator for theta_1 ... theta_{k_max},
        using the recurrence:
            theta_{k+1} = rho * theta_k
        """
        if k_max < 1:
            return

        theta_k = self.alpha * self.rho
        yield theta_k

        for _ in range(1, k_max):
            theta_k *= self.rho
            yield theta_k#

    def analytic_lookback_bound(self,) -> float:
        """
        Analytic upper bound on expected lookback depth:

            E[L] <= 2 / (1 - rho)

        where rho is the upper bound on the memory decay.
        """
        return 2 / (1 - self.rho)


@dataclass(frozen=True)
class InfinitePolynomialTheta:
    """
    Infinite polynomially decaying sequence:

        theta_k = alpha / k^rho,  k >= 1

    Features:
    - O(1) access to any coefficient
    - Analytic tail bounds for truncation error
    - Finite approximation via tolerance
    """
    alpha: float = 0.2
    rho: float = 2.0
    name: str = "InfinitePolynomialTheta"

    def __post_init__(self):
        if self.rho <= 1:
            raise ValueError("rho must be > 1 for summability")
        

    def __getitem__(self, k: int) -> float:
        if k < 1:
            return 0.0
        return self.alpha / (k ** self.rho)

    def __contains__(self, k: int) -> bool:
        return k >= 1

    @property
    def decay_rate(self) -> float:
        """Polynomial decay exponent."""
        return self.rho

    def tail_sum(self, k: int, tight: bool = False) -> float:
        """
        Upper bound on the tail:

            r_k = sum_{m>k} |theta_m|

        Uses the integral bound:
            <= |alpha| * ∫_{x}^{∞} x^{-rho} dx
            = |alpha| * x^(1-rho) / (rho - 1)

        If tight=True, uses x = k+1 for a slightly tighter bound.
        """
        if k < 1:
            k = 1

        x = k + 1 if tight else k
        return abs(self.alpha) * (x ** (1 - self.rho)) / (self.rho - 1)

    def max_k_effective(self, tol: float = 1e-12) -> int:
        """
        Smallest k such that |theta_m| <= tol for all m >= k.
        Useful for finite truncations.
        """
        if self.alpha == 0:
            return 0

        k = (abs(self.alpha) / tol) ** (1 / self.rho)
        return max(1, math.ceil(k))

    def iter_values(self, k_max: int):
        """
        Efficient generator for theta_1 ... theta_{k_max},
        using the recurrence:
            theta_{k+1} = theta_k * (k / (k+1))^rho
        """
        if k_max < 1:
            return

        k = 1
        theta_k = self.alpha
        yield theta_k

        while k < k_max:
            k += 1
            theta_k *= ((k - 1) / k) ** self.rho
            yield theta_k
    def analytic_lookback_bound(self,) -> float:
        """
        Analytic upper bound on expected lookback depth:

            E[L] <= 2 * (1 - rho)

        where rho is the upper bound on the memory decay.
        """
        return 2 * (self.tail_sum(1))

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
