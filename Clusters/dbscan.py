import dist_metrics as dm

class DBSCAN(object):

    """
    Perform DBSCAN clustering.

    Parameters
    ----------
    eps : float, optional
        The maximum distance between two samples for them to be considered
        as in the same neighborhood.

    min_samples : int, optional
        The number of samples in a neighborhood for a point
        to be considered as a core point. This includes the point itself.

    metric : string, optional
        The metric to use when calculating distance between instances in a
        feature array.

    p : float, optional
        The power of the Minkowski metric to be used to calculate distance
        between points.
    """

    def __init__(self, eps=0.5, min_samples=5, metric='euclidean', p=2):

        self.eps = eps
        self.min_sample = min_samples
        self.metric = dm.define_metric(metric, p)


    def _are_neighbors(self, x1, x2):

        """
        Check whether the two samples are neighbors or not.
        """

        distance = self.metric.pairwise_distance(x1, x2)

        if distance <= self.eps:
            return True
        else:
            return False

    def _build_core_cluster(self, X_cores, x_seed):

        """
        Build the cluster of core points corresponding to x_seed sample.
        """

        x_seens = [x_seed]
        x_neighbors = [x_c for x_c in X_cores if self._are_neighbors(x_seed, x_c)]
        x_not_seens = [x_c for x_c in X_cores if x_c not in x_seens and x_c not in x_neighbors]

        for x_n in x_neighbors:
            if x_n not in x_seens:

                for x_c in x_not_seens:
                    if self._are_neighbors(x_n, x_c):
                        x_neighbors.append(x_c)
                        x_not_seens.remove(x_c)

                x_seens.append(x_n)
        
        cluster = x_seens
        return cluster

    def classify_points(self, X):

        """
        Find the core points, border points, and noise points.
        """

        core_points, border_points, noise_points = [], [], []

        #Find core points
        for x_sample in X:

            x_neighbors_counter = 1
            for x_other in X:
                if x_sample != x_other:

                    if self._are_neighbors(x_sample, x_other):

                        x_neighbors_counter += 1
                        if x_neighbors_counter >= self.min_sample:
                            core_points.append(x_sample)
                            break

        #Find border points
        for x_sample in X:
            if x_sample not in core_points:
                for x_core in core_points:

                    if self._are_neighbors(x_sample, x_core):
                        border_points.append(x_sample)
                        break

        #Find noise points
        for x_sample in X:
            if x_sample not in core_points:
                if x_sample not in border_points:
                    noise_points.append(x_sample)

        return core_points, border_points, noise_points


    def predict(self, X):

        X_cores, X_borders, _ = self.classify_points(X)
        clusters = []
        X_cores_not_in_clusters = X_cores

        for x_c in X_cores:
            if x_c in X_cores_not_in_clusters:

                new_cluster = self._build_core_cluster(X_cores_not_in_clusters, x_c)
                clusters.append(new_cluster)
                X_cores_not_in_clusters = [x for x in X_cores_not_in_clusters if x not in new_cluster]


        X_borders_in_cluster = []
        for clus in clusters:
            X_borders = [x_b for x_b in X_borders if x_b not in X_borders_in_cluster]
            for x_b in X_borders:
                for x_c in clus:

                    if self._are_neighbors(x_c, x_b):
                        clus.append(x_b)
                        X_borders_in_cluster.append(x_b)
                        break
        
        return clusters