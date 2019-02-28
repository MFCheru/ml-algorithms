import numpy as np
import math as ma
from abc import ABC, abstractmethod

def define_metric(metric='euclidean', metri_params=2):

    """
    Select and return the metric requested
    """

    if metric == 'euclidean':
        return EuclideanMetric()
    elif metric == 'minkowski':
        return MinkowskiMetric(metri_params)
    else:
        raise ValueError("Invalid metric configured")

def minkowsky(x, y, p=2):

    """
    Distance function: sum(|x - y|^p)^(1/p)
    """

    summatory = 0
    
    for a, b in zip(np.nditer(x), np.nditer(y)):
        summatory  += ma.pow(ma.fabs(a - b), p)

    return ma.pow(summatory,1/p)

def euclidean(x, y):

    """
    Distance function: sqrt(sum((x - y)^2))
    """

    summatory = 0
    
    for a, b in zip(np.nditer(x), np.nditer(y)):
        summatory  += ma.pow(a - b, 2)

    return ma.sqrt(summatory)

class DistanceMetric(ABC):

    @abstractmethod
    def pairwise_distance(self, x, y):

        """
        Calcute the distance between x and y.

        Parameters
        ----------
        x : Array-like, shape (n_samples_a, n_features)
            First sample.

        y : Array-like, shape (n_samples_a, n_features)
            Second sample.
        """
    
        raise NotImplementedError


class MinkowskiMetric(DistanceMetric):
 
    def __init__(self, p=2):
        self.p = p

    def pairwise_distance(self, x, y):
        return minkowsky(x, y, self.p)


class EuclideanMetric(DistanceMetric):

    def pairwise_distance(self, x, y):
        return euclidean(x, y)