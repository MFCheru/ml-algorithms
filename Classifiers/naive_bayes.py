from scipy.stats import norm
from classifier import Classifier

class NaiveBayesClassifier(Classifier):
    """
    A Gaussian Naive Bayes classifier.

    Parameters
    ----------
    priors : array-like, shape (n_classes,)
        Prior probabilities of the classes.
    """

    def __init__(self, priors=None):

        self.means = None
        self.stds = None
        self.priors = None
        self.x = None
        self.y = None


    def _split_by_categories(self, X_train):
        
        X_by_categories = dict([ (cat, []) for cat in set(self.y)])

        for i, x_sample in enumerate(X_train):

            categ = list(self.y)[i]
            X_by_categories[categ].append(x_sample)

        return X_by_categories


    def fit(self, X_train, y_train):

        if len(X_train != y_train):
            raise ValueError("Size of X_train and y_train must be equal")

        self.x = X_train
        self.y = y_train

        n_columns = len(X_train[0])
        means = dict([ (cat, [None]*n_columns) for cat in set(self.y)])
        stds = dict([ (cat, [None]*n_columns) for cat in set(self.y)])

        for cat in set(self.y):
            for i in range(n_columns):

                X_col_cat = [ X_train[j][i] for j in range(len(X_train)) if y_train[j] == cat]
                means[cat][i], stds[cat][i] = norm.fit(X_col_cat)

        self.means = means
        self.stds = stds


    def predict(self, X_test):

        return super().predict(X_test)


    def predict_proba(self, X_test):

        return super().predict_proba(X_test)


    def _predict_sample_proba(self, x_sample):

        if self.means == None or self.stds == None:
            raise ValueError("Model must be fitted previously")

        X_by_categories = self._split_by_categories(self.x)
        x_sample_posteriors = dict()
        categories = set(self.y)
        no_priors = False

        if self.priors == None:
            no_priors = True
            self.priors = dict()

        for category in categories:
            x_sample_posteriors[category] = None

        for category in categories:

            if no_priors:
                self.priors[category] = len(X_by_categories[category]) / len(self.x)

            x_sample_posteriors[category] = self.priors[category]
  
            for j, feature in enumerate(x_sample):
                x_sample_posteriors[category] *= norm.pdf(feature, 
                                                        loc  =self.means[category][j], 
                                                        scale=self.stds[category][j])

        # Calculate the normalization constant
        normalization_constant = 0
        for pos in x_sample_posteriors.values():
            normalization_constant += pos

        # Divide all values by the nomalization constant
        for category in categories:
            x_sample_posteriors[category] /= normalization_constant


        return x_sample_posteriors