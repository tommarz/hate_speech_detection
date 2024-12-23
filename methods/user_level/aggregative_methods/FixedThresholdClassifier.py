import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score

class FixedThresholdClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, proba_threshold=0.5, normalized=True):
        """
        Initialize the classifier.

        Args:
        - proba_threshold: float, threshold to determine if a post is hateful.
        - normalized: bool, whether to use normalized counts for thresholding.
        """
        self.proba_threshold = proba_threshold
        self.normalized = normalized
        self.threshold = None

    def _objective_function(self, threshold, X, y):
        """
        Compute the objective function (negative ROC AUC) for a given threshold.
        """
        # Select the feature based on normalization flag
        feature = 'normalized_hateful_posts' if self.normalized else 'hateful_posts'
        preds = (X[feature] >= threshold).astype(int)
        return -roc_auc_score(y, preds)  # Negative because we want to maximize AUC

    def _optimize_threshold(self, X, y):
        """
        Optimize the threshold using a numerical search method.
        """
        from scipy.optimize import minimize_scalar

        # Define the feature based on normalization
        feature = 'normalized_hateful_posts' if self.normalized else 'hateful_posts'

        # Define bounds for the threshold search
        bounds = (0, 1) if self.normalized else (0, X[feature].max())

        # Minimize the negative ROC AUC
        result = minimize_scalar(
            self._objective_function, bounds=bounds, args=(X, y), method="bounded"
        )
        return result.x  # Optimal threshold

    def fit(self, X, y):
        """
        Fit the classifier to the data by learning the optimal threshold.

        Args:
        - X: DataFrame, input features.
        - y: array-like, target labels.
        """
        # Learn the optimal threshold
        self.threshold = self._optimize_threshold(X, y)
        return self

    def predict(self, X):
        """
        Predict whether each user is a hatemonger.

        Args:
        - X: DataFrame, input features.

        Returns:
        - Array of binary predictions.
        """
        if self.threshold is None:
            raise ValueError("The classifier must be fitted before making predictions.")

        # Select the feature based on normalization flag
        feature = 'normalized_hateful_posts' if self.normalized else 'hateful_posts'
        return (X[feature] >= self.threshold).astype(int)

    def predict_proba(self, X):
        """
        Predict probabilities for each user being a hatemonger.

        Args:
        - X: DataFrame, input features.

        Returns:
        - Array of probabilities for each class.
        """
        preds = self.predict(X)
        return np.vstack([1 - preds, preds]).T

    def get_threshold(self):
        """
        Return the learned threshold.
        """
        return self.threshold