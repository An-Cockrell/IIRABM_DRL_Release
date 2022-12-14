import numpy as np
import copy
"""
Noises processes
    These define how noise is added to the training policy to encourage exploration
"""
class GaussianNoiseProcess:
    """
    Simply adds noise of N(0, std^2)
    """
    def __init__(self, std, shape):
        self.std = std
        self.shape = shape
    def sample(self):
        return np.random.normal(np.zeros(self.shape), self.std)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=0., theta=0.4, sigma=.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.uniform(-1,1) for i in range(len(x))])
        self.state = x + dx
        return self.state
