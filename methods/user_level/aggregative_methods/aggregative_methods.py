import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd
from scipy.special import softmax

class NetworkLevelClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, graph, aggregation='mean'):
        """
        Initialize the classifier.

        Args:
        - graph (igraph.Graph): A directed weighted graph.
        - aggregation (str): Aggregation method ('min', 'max', 'mean', 'sum').
        """
        self.graph = graph
        self.aggregation = aggregation
        self.node_weights = None

    def _aggregate_features(self, node_features, neighbors, weights):
        """
        Aggregate features of neighbors using the specified aggregation method.

        Args:
        - node_features (dict): Features for each node.
        - neighbors (list): List of neighbor node indices.
        - weights (list): Weights for each neighbor.

        Returns:
        - Aggregated feature value.
        """
        neighbor_features = np.array([node_features[n] for n in neighbors])
        weighted_features = neighbor_features * np.array(weights).reshape(-1, 1)

        if self.aggregation == 'mean':
            return weighted_features.mean(axis=0)
        elif self.aggregation == 'sum':
            return weighted_features.sum(axis=0)
        elif self.aggregation == 'max':
            return weighted_features.max(axis=0)
        elif self.aggregation == 'min':
            return weighted_features.min(axis=0)
        else:
            raise ValueError("Invalid aggregation method. Use 'mean', 'sum', 'max', or 'min'.")

    def fit(self, X, y):
        """
        Fit the classifier by learning the optimal weights for neighbors.

        Args:
        - X (dict): Features for each node (node ID -> feature vector).
        - y (dict): Labels for each node (node ID -> label).
        """
        self.node_weights = np.ones(self.graph.vcount())  # Initialize weights for each node

        def objective(weights):
            total_loss = 0
            for v in range(self.graph.vcount()):
                neighbors = self.graph.neighbors(v, mode='in')  # Incoming edges
                edge_weights = [self.graph.es[self.graph.get_eid(n, v)]['weight'] for n in neighbors]
                aggregated_feature = self._aggregate_features(X, neighbors, edge_weights * weights)
                total_loss += np.sum((aggregated_feature - X[v]) ** 2)  # Example: Mean squared error
            return total_loss

        result = minimize(objective, self.node_weights, method='BFGS')
        self.node_weights = result.x

        return self

    def predict(self, X):
        """
        Predict the labels for each node.

        Args:
        - X (dict): Features for each node (node ID -> feature vector).

        Returns:
        - Array of binary predictions (0 or 1).
        """
        predictions = []
        for v in range(self.graph.vcount()):
            neighbors = self.graph.neighbors(v, mode='in')  # Incoming edges
            edge_weights = [self.graph.es[self.graph.get_eid(n, v)]['weight'] for n in neighbors]
            aggregated_feature = self._aggregate_features(X, neighbors, edge_weights * self.node_weights)
            predictions.append(aggregated_feature)  # Replace with a final decision rule if needed
        return np.array(predictions)

    def predict_proba(self, X):
        """
        Predict probabilities for each node.

        Args:
        - X (dict): Features for each node (node ID -> feature vector).

        Returns:
        - Array of probabilities for each class.
        """
        # Implement if probability prediction is needed
        pass

    def transform(self, X):
        """
        Transform the input features by aggregating the features of each node and its neighbors.

        Args:
        - X (dict): Features for each node (node ID -> feature vector).

        Returns:
        - dict: Transformed features for each node (node ID -> aggregated feature vector).
        """
        transformed_features = {}
        for v in range(self.graph.vcount()):
            neighbors = self.graph.neighbors(v, mode='in')  # Incoming edges
            edge_weights = [self.graph.es[self.graph.get_eid(n, v)]['weight'] for n in neighbors]
            aggregated_feature = self._aggregate_features(X, neighbors, edge_weights * self.node_weights)
            transformed_features[v] = aggregated_feature
        return transformed_features

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
        Fit the classifier by learning the optimal threshold.

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
        Return probabilities for each user being a hatemonger.
        """
        preds = self.predict(X)
        return np.vstack([1 - preds, preds]).T

    def transform(self, X):
        """
        Transform the input data by computing hateful post counts.

        Args:
        - X: DataFrame, input features.

        Returns:
        - DataFrame: Transformed features with hateful post counts.
        """
        if self.threshold is None:
            raise ValueError("The classifier must be fitted before transforming data.")

        feature = 'normalized_hateful_posts' if self.normalized else 'hateful_posts'
        X['hateful_posts'] = (X[feature] >= self.threshold).astype(int)
        X['total_hateful_posts'] = X.groupby('username')['hateful_posts'].transform('sum')
        return X

class DistributionalAggregatorClassifier(BaseEstimator, ClassifierMixin):
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

    def transform(self, X):
        """
        Transform the input data by aggregating the hate scores into features using learned weights.

        Args:
        - X (DataFrame): Data containing usernames and hate scores.

        Returns:
        - DataFrame: Transformed features including bin counts and aggregated scores.
        """
        user_scores = X.groupby('username')['hate_scores'].apply(list)
        transformed_features = []
        for username, hate_scores in user_scores.items():
            bin_counts = self._bin_counts(hate_scores)
            softmax_weights = softmax(self.weights)
            aggregated_feature = np.dot(softmax_weights, bin_counts)
            transformed_features.append({
                'username': username,
                'aggregated_feature': aggregated_feature,
                'bin_counts': bin_counts.tolist(),
                'total_hateful_posts': sum(hate_scores)
            })
        return pd.DataFrame(transformed_features)