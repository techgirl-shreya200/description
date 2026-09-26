import numpy as np


def bce_loss(y: np.ndarray, yhat: np.ndarray) -> float:
    n = y.shape[0]
    return -(np.sum(y * np.log(yhat) + (1 - y) * np.log(1 - yhat)) / n)
