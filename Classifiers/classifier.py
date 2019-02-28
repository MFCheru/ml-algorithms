from abc import ABC, abstractmethod

class Classifier(ABC):

    @abstractmethod
    def fit(self, X_train):

        """
        Infers a model based on the X_train dataset
        """

        raise NotImplementedError


    def predict(self, X_test):

        """
        Calculate most likely class corresponding to each sample of X_test 
        """

        y_pred = []
        y_test_proba_predicted = self.predict_proba(X_test)

        for y_sample_proba_predicted in y_test_proba_predicted:

            max_proba = 0
            for category in y_sample_proba_predicted.keys():

                proba = y_sample_proba_predicted[category]

                if max_proba < proba:
                    max_proba = proba
                    max_category = category

            y_pred.append(max_category)
        
        return y_pred


    def predict_proba(self, X_test):

        """
        Calculate the probability for each sample of X_test to belongs to each class
        """

        predictions = []

        for x_sample in X_test:

            prediction = self._predict_sample_proba(x_sample)
            predictions.append(prediction)

        return predictions

    @abstractmethod
    def _predict_sample_proba(self, x_sample):

        """
        Calculate the probability for x_sample to belongs to each class
        """

        raise NotImplementedError

    def score(self, X_test, y_test):

        """
        Score the fitted model
        """

        y_pred = self.predict(X_test)
        accuracy = sum(y_i == y_j for (y_i, y_j) in zip(y_pred, y_test)) / len(y_test)

        return accuracy