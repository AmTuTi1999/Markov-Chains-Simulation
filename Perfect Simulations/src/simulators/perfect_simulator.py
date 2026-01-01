from abc import ABC, abstractmethod
class PerfectSimulator(ABC):
    """
    Generic base class for perfect simulation algorithms
    (CFTP, regenerative schemes, Fill’s method, etc.)
    """

    @abstractmethod
    def perfect_sample(self, n=0):
        """
        Return a perfect sample X_n from the stationary distribution.
        Must be implemented by subclasses.
        """
        pass

    def simulate_window(self, s, t):
        """
        Default window simulation by calling perfect_sample() repeatedly.
        Child classes may override with more efficient schemes.
        """
        return [self.perfect_sample(n) for n in range(s, t + 1)]