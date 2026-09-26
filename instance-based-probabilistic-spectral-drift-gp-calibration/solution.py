import numpy as np

R_H = 1.097373e7  # m^-1

def rydberg_wavelength_nm(n1: int, n2: int) -> float:
    return 1e9 / (R_H * (1 / n1**2 - 1 / n2**2))

def gp_calibrate(
    training: list[tuple[int, int, float]],
    sigma_f: float,
    l: float,
    sigma_n: float,
    queries: list[tuple[int, int]],
) -> np.ndarray:
    X = np.array([rydberg_wavelength_nm(n1, n2) for n1, n2, _ in training])
    y = np.array([measured - rydberg_wavelength_nm(n1, n2) for n1, n2, measured in training])
    n = len(X)
    K = sigma_f**2 * np.exp(-(X[:, None] - X[None, :])**2 / (2 * l**2))
    A = K + sigma_n**2 * np.eye(n)
    L = np.linalg.cholesky(A)
    z = np.linalg.solve(L, y)
    alpha = np.linalg.solve(L.T, z)
    results = []
    for q_n1, q_n2 in queries:
        x_star = rydberg_wavelength_nm(q_n1, q_n2)
        k_star = sigma_f**2 * np.exp(-(X - x_star)**2 / (2 * l**2))
        mu = k_star @ alpha
        v = np.linalg.solve(L, k_star)
        sigma2 = sigma_f**2 - v @ v
        results.append([x_star + mu, np.sqrt(max(sigma2, 0))])
    return np.array(results)
