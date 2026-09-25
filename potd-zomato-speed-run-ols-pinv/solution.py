import numpy as np


def ols_fit_predict(X: np.ndarray, y: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
    Q_aug = np.hstack([np.ones((Q.shape[0], 1)), Q])
    w = np.linalg.pinv(X_aug) @ y
    predictions = Q_aug @ w
    return w, predictions
