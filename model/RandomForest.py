import numpy as np
from joblib import Parallel, delayed
try:
    from model.DecisionTree import RegressionTree
except ImportError:
    from DecisionTree import RegressionTree


def _train_single_tree(X, y, max_depth, min_samples_split, n_features, seed):
    """Train một cây riêng lẻ - đặt ngoài class để joblib pickle được"""
    np.random.seed(seed)

    n_samples = X.shape[0]
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    X_sample, y_sample = X[indices], y[indices]

    tree = RegressionTree(
        min_samples_split=min_samples_split,
        max_depth=max_depth,
        n_features=n_features
    )
    tree.train(X_sample, y_sample)
    return tree


def _predict_single_tree(tree, X):
    """Predict với một cây - đặt ngoài class để joblib pickle được"""
    return tree.predict(X)


class RandomForest:
    def __init__(self, n_trees=100, max_depth=10, min_samples_split=2,
                 n_features=None, random_state=None, n_jobs=-1):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.trees = []

    def fit(self, X, y):
        # Tạo random generator riêng để không ảnh hưởng global state
        rng = np.random.RandomState(self.random_state)
        seeds = rng.randint(0, 2 ** 31 - 1, self.n_trees)

        n_features = self.n_features
        if n_features is None:
            n_features = max(1, int(X.shape[1] / 3))

        # Chạy song song với verbose để theo dõi tiến trình
        self.trees = Parallel(n_jobs=self.n_jobs, verbose=0)(
            delayed(_train_single_tree)(
                X, y,
                self.max_depth,
                self.min_samples_split,
                n_features,
                seed
            ) for seed in seeds
        )

    def predict(self, X):
        # Với predict, chỉ nên dùng parallel nếu có nhiều cây
        # Vì overhead của parallel có thể làm chậm hơn với ít cây
        if self.n_trees >= 50 and len(X) >= 1000:
            tree_preds = Parallel(n_jobs=self.n_jobs)(
                delayed(_predict_single_tree)(tree, X) for tree in self.trees
            )
        else:
            tree_preds = [tree.predict(X) for tree in self.trees]

        return np.mean(tree_preds, axis=0)