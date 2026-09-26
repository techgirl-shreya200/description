import numpy as np

def circuit_output(x: np.ndarray, theta: np.ndarray, n: int, L: int) -> float:
    state = np.zeros(2**n)
    state[0] = 1.0
    def apply_ry(state, angle, qubit, n):
        cos = np.cos(angle/2)
        sin = np.sin(angle/2)
        new_state = state.copy()
        for i in range(2**n):
            if (i >> qubit) & 1 == 0:
                j = i | (1 << qubit)
                a, b = state[i], state[j]
                new_state[i] = cos * a - sin * b
                new_state[j] = sin * a + cos * b
        return new_state
    def apply_cnot(state, control, target, n):
        new_state = state.copy()
        for i in range(2**n):
            if (i >> control) & 1:
                j = i ^ (1 << target)
                new_state[j] = state[i]
        return new_state
    for i in range(n):
        state = apply_ry(state, x[i], i, n)
    for l in range(L):
        for i in range(n-1):
            state = apply_cnot(state, i, i+1, n)
        for i in range(n):
            state = apply_ry(state, theta[l, i], i, n)
    prob0 = sum(abs(state[i])**2 for i in range(2**n) if (i & 1) == 0)
    prob1 = 1 - prob0
    return prob0 - prob1

def train_and_predict(
    X: np.ndarray,
    y: np.ndarray,
    n: int,
    L: int,
    theta_init: np.ndarray,
    eta: float,
    T: int,
    queries: np.ndarray,
) -> np.ndarray:
    theta = np.array(theta_init).reshape(L, n)
    for _ in range(T):
        preds = np.array([circuit_output(x, theta, n, L) for x in X])
        errors = preds - y
        grad = np.zeros_like(theta)
        for l in range(L):
            for i in range(n):
                shift = np.zeros_like(theta)
                shift[l, i] = np.pi/2
                plus = np.array([circuit_output(x, theta+shift, n, L) for x in X])
                minus = np.array([circuit_output(x, theta-shift, n, L) for x in X])
                dps = (plus - minus) / 2
                grad[l, i] = (2/len(X)) * np.sum(errors * dps)
        theta -= eta * grad
    return np.array([circuit_output(q, theta, n, L) for q in queries])
