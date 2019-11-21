import numpy as np

# Collection of usual kernels
class Kernel(object):
    @staticmethod
    def linear():
        def f(x, y):
            return np.inner(x, y)
        return f

    @staticmethod
    def rbf(gamma):
        def f(x, y):
            return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * (gamma ** 2)))
            #exponent = - gamma * np.linalg.norm(x-y) ** 2
            #return np.exp(exponent)
        return f

    @staticmethod
    def _polykernel(dimension, offset=0.0, gamma=1.0):
        def f(x, y):
            return (gamma * (offset + np.dot(x, y)) )** dimension
        return f

    @staticmethod
    def quadratic(offset=0.0, gamma=1.0):
        def f(x, y):
            return (gamma * (offset + np.dot(x, y)) )** 2
        return f