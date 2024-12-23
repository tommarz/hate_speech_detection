import numpy as np
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score

class DistributionalAggregationClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, k=10, method='bin', threshold=0.5):
        """
        Initialize the classifier.

        Args:
        - k (int): Number of bins or quantiles.
        - method (str): 'bin' for bin-based, 'quantile' for quantile-based aggregation.
        - threshold (float): Threshold for classification.
        """
        self.k = k
        self.method = method
        self.threshold = threshold
        self.weights = None

    def _bin_counts(self, hate_scores):
        """Compute bin counts based on the selected method (bin or quantile)."""
        if self.method == 'bin':
            # Equal-width bins in [0, 1]
            bin_edges = np.linspace(0, 1, self.k + 1)
        elif self.method == 'quantile':
            # Quantile-based bins
            quantiles = np.linspace(0, 1, self.k + 1)
            bin_edges = np.quantile(hate_scores, quantiles)
        else:
            raise ValueError("Invalid method. Use 'bin' or 'quantile'.")

        # Count the number of hate scores in each bin
        bin_counts, _ = np.histogram(hate_scores, bins=bin_edges)
        return bin_counts

    def fit(self, X, y):
        """
        Fit the classifier by learning the optimal weights for the bins.

        Args:
        - X (array-like): Hate scores for posts.
        - y (array-like): Binary labels for the users (0 or 1).
        """
        # Aggregate hate scores for each user
        user_scores = X.groupby('username')['hate_scores'].apply(list)

        # Initialize weights
        self.weights = np.ones(self.k)

        # Optimize weights using ROC AUC
        def objective(weights):
            total_scores = []
            for hate_scores in user_scores:
                bin_counts = self._bin_counts(hate_scores)
                softmax_weights = softmax(weights)
                total_scores.append(np.dot(softmax_weights, bin_counts))
            return -roc_auc_score(y, total_scores)

        from scipy.optimize import minimize
        result = minimize(objective, self.weights, method='BFGS')
        self.weights = result.x

        return self

    def predict(self, X):
        """
        Predict whether each user is classified as hateful based on the learned weights.

        Args:
        - X (DataFrame): Data containing usernames and hate scores.

        Returns:
        - Array of binary predictions (0 or 1).
        """
        user_scores = X.groupby('username')['hate_scores'].apply(list)
        predictions = []
        for hate_scores in user_scores:
            bin_counts = self._bin_counts(hate_scores)
            softmax_weights = softmax(self.weights)
            aggregated_score = np.dot(softmax_weights, bin_counts)
            predictions.append(1 if aggregated_score >= self.threshold else 0)
        return np.array(predictions)

    def predict_proba(self, X):
        """
        Predict probabilities for each user.

        Args:
        - X (DataFrame): Data containing usernames and hate scores.

        Returns:
        - Array of probabilities for each class.
        """
        user_scores = X.groupby('username')['hate_scores'].apply(list)
        probabilities = []
        for hate_scores in user_scores:
            bin_counts = self._bin_counts(hate_scores)
            softmax_weights = softmax(self.weights)
            aggregated_score = np.dot(softmax_weights, bin_counts)
            probabilities.append(aggregated_score)
        probabilities = np.array(probabilities)
        return np.vstack([1 - probabilities, probabilities]).T
