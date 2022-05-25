import numpy as np

class OUNoise:
    """The Ornstein-Uhlenbeck process generating the noise for exploration"""
    
    def __init__(self, output_size=4, mu=0.0, theta=0.15, sigma=0.20):
        
        """
        output_size: dimension of the noise, which should be the dimension of the action
        mu: the asymptotic mean of the noise
        theta: the magnitude of the drift component
        sigma: the magnitude of the diffusion (Gaussian noise) component
        """
               
        self.output_size = output_size
        self.mu = mu*np.ones(output_size)
        self.theta = theta
        self.sigma = sigma

        self.reset()

    def reset(self):
        """Set the current noise value to the asymptotic mean value"""
        
        self.x = np.copy(self.mu)

    def get_noise(self):
        """Generate a noise vector of dimension = self.output_size"""
        
        self.x += self.theta*(self.mu - self.x) + self.sigma*np.random.randn(self.output_size)
        return self.x