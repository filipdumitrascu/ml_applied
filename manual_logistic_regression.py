import numpy as np

class ManualLogisticRegression:
    def __init__(self, lr=0.1, epochs=3000, l2_lambda=0.01):
        self.lr = lr
        self.epochs = epochs
        self.l2_lambda = l2_lambda
        self.classifiers = {}  # model pentru fiecare clasă
        self.classes_ = None

    def _logistic(self, x):
        return 1 / (1 + np.exp(-x))

    def _nll(self, Y, T, w):
        Y = np.clip(Y, 1e-9, 1 - 1e-9)
        base_loss = -np.mean(T * np.log(Y) + (1 - T) * np.log(1 - Y))
        l2_term = 0.5 * self.l2_lambda * np.sum(w ** 2)
        return base_loss + l2_term

    def _train_binary_classifier(self, X, T):
        N, D = X.shape
        w = np.random.randn(D)

        for _ in range(self.epochs):
            Y = self._logistic(X @ w)
            grad = (1. / N) * X.T @ (Y - T) + self.l2_lambda * w
            w -= self.lr * grad

        return w

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for c in self.classes_:
            T_binary = (y == c).astype(int)
            w = self._train_binary_classifier(X, T_binary)
            self.classifiers[c] = w

    def predict(self, X):
        scores = []
        for c in self.classes_:
            w = self.classifiers[c]
            proba = self._logistic(X @ w)
            scores.append(proba)

        scores = np.array(scores)  # shape (n_classes, n_samples)
        predictions = np.argmax(scores, axis=0)
        return self.classes_[predictions]
