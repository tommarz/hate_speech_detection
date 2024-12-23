import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.optimize import minimize

class RelationalAggregation(BaseEstimator, ClassifierMixin):
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
        - Array of predictions for each node.
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
