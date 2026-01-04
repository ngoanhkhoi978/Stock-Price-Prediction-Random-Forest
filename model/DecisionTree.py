from typing import Tuple, Optional
import numpy as np
import numpy.typing as npt

class Node:
    def __init__(self,
             feature_: int, threshold_: float,
             left_subtree_, right_subtree_):
        self.feature = feature_; self.threshold = threshold_
        self.left_subtree = left_subtree_; self.right_subtree = right_subtree_

def mse(y):
    if len(y) == 0: return 0
    return np.mean((y - np.mean(y)) ** 2)

def best_split(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    n_features_to_select: Optional[int] = None
) -> Tuple[Optional[int], Optional[float], float]:
    best_feature, best_thresh, best_mse = None, None, float("inf")
    n_samples, n_total_features = X.shape

    if n_features_to_select is None:
        feature_indices = range(n_total_features)
    else:
        feature_indices = np.random.choice(n_total_features, n_features_to_select, replace=False)

    for feature in feature_indices:
        thresholds = np.unique(X[:, feature])
        for thresh in thresholds:
            left_bitmask = X[:, feature] <= thresh
            right_bitmask = ~left_bitmask

            if len(y[left_bitmask]) == 0 or len(y[right_bitmask]) == 0:
                continue

            y_left, y_right = y[left_bitmask], y[right_bitmask]

            mse_left = mse(y_left)
            mse_right = mse(y_right)

            n_l, n_r = len(y_left), len(y_right)
            n_total = n_l + n_r
            weighted_mse = (n_l * mse_left + n_r * mse_right) / n_total

            if weighted_mse < best_mse:
                best_feature, best_thresh, best_mse = feature, thresh, weighted_mse

    return best_feature, best_thresh, best_mse


class RegressionTree:
    def __init__(self, max_depth = 5, min_samples_split = 2, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.tree = None

    def fit(self,
            X: npt.NDArray[np.float64],
            y: npt.NDArray[np.float64], depth_ = 0):
        if depth_ >= self.max_depth or len(y) < self.min_samples_split:
            return np.mean(y)

        feature, thresh, current_loss = best_split(X, y, self.n_features)
        if feature is None:
            return np.mean(y)

        left_bitmask = X[:, feature] <= thresh
        right_bitmask = X[:, feature] > thresh

        left_subtree = self.fit(X[left_bitmask], y[left_bitmask], depth_ + 1)
        right_subtree = self.fit(X[right_bitmask], y[right_bitmask], depth_ + 1)

        return Node(feature, thresh, left_subtree, right_subtree)

    def train(self,
              X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]):
        if self.n_features is None:
            self.n_features = int(X.shape[1] / 3) if X.shape[1] > 3 else X.shape[1]

        self.tree = self.fit(X, y)

    def predict_one(self, x, node):
        if not isinstance(node, Node):
            return node
        if x[node.feature] <= node.threshold:
            return self.predict_one(x, node.left_subtree)
        else:
            return self.predict_one(x, node.right_subtree)

    def predict(self, X):
        return np.array([self.predict_one(sample, self.tree) for sample in X])
