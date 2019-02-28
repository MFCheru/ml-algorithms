import numpy as np
import dist_metrics as dm


class KMeans(object):
    """
    Performs K-Means clustering.

    Parameters
    ----------
    n_clusters : integer
        The number of clusters
    
    n_iter : integer
        Number of times that the clusters will be computed

    metric : string, optional (default = 'euclidean')
        The metric to use when calculating distance between instances in a
        feature array.

    p : float, optional
        The power of the Minkowski metric to be used to calculate distance
        between points.
    """

    def __init__(self, n_clusters=2, n_iter=10, metric='euclidean', p=2):

        self.n_clusters = n_clusters
        self.n_iter = n_iter
        self.metric = dm.define_metric(metric, p)
        self.centroids = np.ndarray(0)


    def  _initialize_centroids(self, X):

        """
        Initialize centroids using K-Means++ technique
        """
        self.centroids = np.ndarray(0)
        indicies = np.random.choice(len(X))
        self.centroids = np.append(self.centroids, X[indicies])
        x_no_centroid = np.setdiff1d(X, self.centroids)

        for i in range(self.n_clusters - 1):

                distances_to_centroid = [ self.metric.pairwise_distance(self.centroids[i], x_sample) for x_sample in x_no_centroid ]
                sum_of_distances = sum(distances_to_centroid)

                probabilities = [ dist / sum_of_distances for dist in distances_to_centroid ]
                indicies = np.random.choice(len(x_no_centroid), p=probabilities)
                new_centroid = x_no_centroid[indicies]
                self.centroids = np.append(self.centroids, new_centroid)

                x_no_centroid = np.setdiff1d(x_no_centroid, self.centroids)


    #Calculates the centroids
    def fit(self, X):

        self._initialize_centroids(X)

        for _ in range(self.n_iter):

            clusters = self.predict(X)

            for j in len(self.centroids):

                sum_cluster = sum(clusters[j])
                self.centroids[j] = sum_cluster / len(clusters[j])
                

    #Return an array with the closest cluster each sample in X belongs to
    def predict(self, X):

        clusters = [[] for i in range(len(self.centroids))]
        x_centroid_idx = 0

        for x_sample in X:

            x_centroid_idx = 0
            min_dist_to_centroid = self.metric.pairwise_distance(self.centroids[0], x_sample)

            for j, cent in enumerate(self.centroids):

                x_distance_to_cent =  self.metric.pairwise_distance(cent, x_sample)

                if min_dist_to_centroid > x_distance_to_cent:
                    x_centroid_idx = j 
                
            clusters[x_centroid_idx].append(x_sample)

        return clusters