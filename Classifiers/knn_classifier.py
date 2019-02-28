import dist_metrics as dm
from classifier import Classifier


class KnnClassifier(Classifier):
    """
    k-nearest neighbors classifier.

    Parameters
    ----------
    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default.

    p : integer, optional (default = 2)
        Power parameter for the Minkowski metric.

    metric : string, default 'minkowski'
        The distance metric to use.

    weighted : boolean, (default = False)
        Indicates if votes of the neighbors are counted uniformly or as 1 / distance
    """

    def __init__(self, n_neighbors=5, p=2, metric='minkowski', weight=False):

        self.n_neighbors = n_neighbors
        self.weight = weight
        self.x = []
        self.y = []
        self.metric = dm.define_metric(metric, p)


    def fit(self, X_train, y_train):

        if len(X_train != y_train):
            raise ValueError("Size of X_train and y_train must be equal")

        self.x = X_train
        self.y = y_train


    def _neighbors(self, x_sample):

        """
        Return the indicies of the n nearest neighbors of x_sample
        """

        x_knn = []
        x_distance = 0
        max_dist = 0

        for idx, x_dev in enumerate(self.x):

            x_distance = self.metric.pairwise_distance(x_sample, x_dev)

            if len(x_knn) < self.n_neighbors:

                x_knn.append((idx, x_distance))
                max_dist = max(x_knn, key=lambda item:item[1])[1]

            elif max_dist > x_distance:

                    x_knn.remove(max(x_knn, key=lambda item:item[1]))
                    x_knn.append((idx, x_distance))
                    max_dist = max(x_knn, key=lambda item:item[1])[1]
                
        x_knn_idxs = [ idx for (idx, x_distance) in x_knn]

        return x_knn_idxs


    def predict(self, X_test):

        return super().predict(X_test)


    def predict_proba(self, X_test):

        return super().predict_proba(X_test)


    def _predict_sample_proba(self, x_sample):

        x_knn_idxs = self._neighbors(x_sample)
        x_votes = dict()
        categories = set(self.y)

        for category in categories:
            x_votes[category] = 0

        weights_sum = 0
        for idx in x_knn_idxs:

            category = self.y[idx]

            if self.weight:
                d = self.metric.pairwise_distance(x_sample, self.x[idx])
                if d==0: d = 1
                vote = 1/d     
            else:
                vote = 1
                
            x_votes[category] += vote
            weights_sum += vote   

        y_proba_predicted = dict()
        for category in categories:

            x_posteriori = x_votes[category] / weights_sum
            y_proba_predicted[category] = x_posteriori
        
        return y_proba_predicted