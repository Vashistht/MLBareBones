"""18-661 HW5 Optimization Policies."""

import numpy as np

from .base import Optimizer


class SGD(Optimizer):
    """Simple SGD optimizer.

    Parameters
    ----------
    learning_rate : float
        SGD learning rate.
    """

    def __init__(self, learning_rate=0.01):

        self.learning_rate = learning_rate

    def apply_gradients(self, params):
        """Apply gradients to parameters.

        Parameters
        ----------
        params : Variable[]
            List of parameters that the gradients correspond to.
        """
        # raise NotImplementedError()
        # w = w - lr * grad
        for i in range(len(params)):
            params[i].value = params[i].value - self.learning_rate * params[i].grad


class Adam(Optimizer):
    """Adam (Adaptive Moment) optimizer.

    Parameters
    ----------
    learning_rate : float
        Learning rate multiplier.
    beta1 : float
        Momentum decay parameter.
    beta2 : float
        Variance decay parameter.
    epsilon : float
        A small constant added to the demoniator for numerical stability.
    """

    def __init__(
            self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7):

        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.tstep = 0

    def initialize(self, params):
        """Initialize any optimizer state needed.

        params : np.array[]
            List of parameters that will be used with this optimizer.
        """
        # raise NotImplementedError()÷
        self.m = np.array([])
        self.v = np.array([])
        for param in params:
            self.m = np.append(self.m, np.zeros(param.value.shape))
            self.v = np.append(self.v, np.zeros(param.value.shape))
        

    def apply_gradients(self, params):
        """Apply gradients to parameters.

        Parameters
        ----------
        params : Variable[]
            List of parameters that the gradients correspond to.
        """
        # raise NotImplementedError()
        # mt = β1mt−1 + (1 − β1)gt
        # vt = β2vt−1 + (1 − β2)gt**2
        # m = mt / 1−βt1
        # v = vt/ 1−βt2
        
        self.tstep += 1
        for i in range(len(params)):
            self.m[i] = (self.beta1 * self.m[i]) + (1 - self.beta1) * params[i].grad
            self.v[i] = (self.beta2 * self.v[i]) + (1 - self.beta2) * params[i].grad**2
            m_hat = self.m[i] / (1 - self.beta1**self.tstep) # update m and v
            v_hat = self.v[i] / (1 - self.beta2**self.tstep)
            
            # update the theta value
            params[i].value = params[i].value - (self.learning_rate * m_hat) / (np.sqrt(v_hat) + self.epsilon)
