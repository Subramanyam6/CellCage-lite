"""Lightweight classifier heads that sit on top of frozen DINOv2 embeddings.

The embeddings do the hard work, so the head can stay simple. Two are provided,
matching the README:

- :class:`NearestClassMean` (a prototypical classifier): label a cell by the
  closest average embedding per class. Needs no training, so it recognizes a new
  cell type from a handful of examples.
- :class:`LinearProbe`: a regularized multinomial logistic regression. Trains in
  seconds, needs few examples, and will not overfit.

Both are pure NumPy so they run without a deep-learning stack; only the embedding
step (``classify/embed.py``) needs torch.
"""

from __future__ import annotations

import numpy as np


class NearestClassMean:
    """Prototypical classifier: assign the label of the nearest class centroid.

    Distances are computed on L2-normalized embeddings, so "nearest" is by cosine
    similarity, which is what DINOv2 features are usually compared with.
    """

    def __init__(self) -> None:
        self.classes_: np.ndarray | None = None
        self._centroids: np.ndarray | None = None

    def fit(self, embeddings: np.ndarray, labels: np.ndarray) -> "NearestClassMean":
        embeddings = _l2_normalize(np.asarray(embeddings, dtype=float))
        labels = np.asarray(labels)
        self.classes_ = np.unique(labels)
        self._centroids = np.stack(
            [_l2_normalize(embeddings[labels == c].mean(axis=0, keepdims=True))[0]
             for c in self.classes_]
        )
        return self

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        if self._centroids is None or self.classes_ is None:
            raise RuntimeError("call fit() before predict()")
        x = _l2_normalize(np.asarray(embeddings, dtype=float))
        similarity = x @ self._centroids.T  # cosine, since both are normalized
        return self.classes_[np.argmax(similarity, axis=1)]


class LinearProbe:
    """Regularized multinomial logistic regression trained by gradient descent.

    Inputs are standardized before training so the single learning rate behaves
    across features. Small, L2-regularized, and quick to fit.
    """

    def __init__(self, l2: float = 1e-2, lr: float = 0.5, epochs: int = 300) -> None:
        self.l2 = l2
        self.lr = lr
        self.epochs = epochs
        self.classes_: np.ndarray | None = None
        self._W: np.ndarray | None = None
        self._b: np.ndarray | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def fit(self, embeddings: np.ndarray, labels: np.ndarray) -> "LinearProbe":
        x = np.asarray(embeddings, dtype=float)
        self._mean = x.mean(axis=0, keepdims=True)
        self._std = x.std(axis=0, keepdims=True) + 1e-8
        x = (x - self._mean) / self._std

        labels = np.asarray(labels)
        self.classes_ = np.unique(labels)
        k = len(self.classes_)
        class_index = {c: i for i, c in enumerate(self.classes_)}
        y = np.array([class_index[c] for c in labels])
        y_onehot = np.eye(k)[y]

        n, d = x.shape
        self._W = np.zeros((d, k))
        self._b = np.zeros(k)
        for _ in range(self.epochs):
            probs = _softmax(x @ self._W + self._b)
            grad = probs - y_onehot
            self._W -= self.lr * (x.T @ grad / n + self.l2 * self._W)
            self._b -= self.lr * grad.mean(axis=0)
        return self

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        if self._W is None:
            raise RuntimeError("call fit() before predict()")
        x = (np.asarray(embeddings, dtype=float) - self._mean) / self._std
        return _softmax(x @ self._W + self._b)

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("call fit() before predict()")
        return self.classes_[np.argmax(self.predict_proba(embeddings), axis=1)]


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
