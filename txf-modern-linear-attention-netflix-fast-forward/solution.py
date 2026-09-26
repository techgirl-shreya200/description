import numpy as np

def feature_map(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0) + 1

def linear_attention_row_sums(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    Qp = feature_map(Q)
    Kp = feature_map(K)
    M = Kp.T @ V
    Z = np.sum(Kp, axis=0)
    numerator = Qp @ M
    denominator = Qp @ Z
    return np.sum(numerator, axis=1) / denominator
