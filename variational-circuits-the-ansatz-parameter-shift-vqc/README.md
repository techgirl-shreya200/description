# THE ANSATZ

Advanced | quantum-computing | quantum-machine-learning | optimization | physics

**Difficulty:** Hard
**Topic:** Variational Quantum Circuits, Parameter-Shift Gradients, Quantum Machine Learning

---

### Story

You're an intern at QuantEdge Capital, a fund experimenting with early quantum hardware for
trading-signal classification. Their prototype device only reliably runs a handful of qubits,
so the research team settled on a **hardware-efficient variational quantum circuit**: encode
each market feature as a single-qubit rotation, entangle the qubits, apply a small number of
trainable rotation layers, and read out one qubit's expectation value as the predicted signal.

Real quantum hardware is expensive lab time, so before touching the device you're asked to
build the **classical simulator and trainer** first: simulate the circuit's exact quantum
state, compute gradients using the _parameter-shift rule_ (the standard way to differentiate
a quantum circuit — you cannot backpropagate through real hardware, but this identity lets you
get an exact gradient using only two extra circuit evaluations per parameter), and train the
circuit's parameters with gradient descent.

---

### The circuit

You work with `n` qubits (`1 ≤ n ≤ 4`), state space dimension `2^n`, initialized to
`|0...0⟩`. All gates used are **real-valued**, so the state vector's amplitudes stay real
throughout — no complex arithmetic is ever required (this is exactly the "RealAmplitudes"
family of ansätze used in real near-term quantum ML). Qubits are indexed `0 .. n-1`.

Two gates are used:

```
RY(θ) = [[cos(θ/2), -sin(θ/2)],
         [sin(θ/2),  cos(θ/2)]]     (acts on one qubit)

CNOT(control, target): if control qubit is |1⟩, flip target qubit; otherwise do nothing.
```

**Fixed circuit structure**, given a feature vector `x ∈ ℝ^n` and trainable parameters
`θ`, organized as `L` layers of `n` angles each (`θ_{l,i}`, `l = 0..L-1`, `i = 0..n-1`):

1. **Encoding layer:** for `i = 0 .. n-1`, apply `RY(x_i)` to qubit `i`.
2. For each layer `l = 0 .. L-1`:
   a. **Entangling chain:** for `i = 0 .. n-2`, apply `CNOT(control=i, target=i+1)`, in
   increasing order of `i`.
   b. **Rotation layer:** for `i = 0 .. n-1`, apply `RY(θ_{l,i})` to qubit `i`.

(If `n = 1` the entangling chain is empty — there is nothing to entangle with one qubit.)

**Measurement / model output:** the prediction is the exact expectation value of the Pauli-Z
operator on qubit `0`:

```
p(x, θ) = ⟨ψ| Z⊗I⊗...⊗I |ψ⟩ = P(qubit 0 = |0⟩) - P(qubit 0 = |1⟩)
```

where `|ψ⟩` is the final state after the full circuit above. `p(x, θ) ∈ [-1, 1]` always.

---

### Training

You're given `n_train` labeled examples `(x_i, y_i)`, `y_i ∈ ℝ` (targets are **not**
restricted to `[-1,1]` even though the model's output is — this is an ordinary regression
setup with mean squared error loss):

```
Loss(θ) = (1 / n_train) * Σ_i ( p(x_i, θ) - y_i )^2
```

For any parameter `θ_{l,i}` that enters the circuit only through an `RY` rotation, the
**exact** gradient of the output (no approximation, no finite-difference error) is given by
the parameter-shift rule:

```
∂p(x,θ) / ∂θ_{l,i} = [ p(x, θ + (π/2)·e_{l,i}) - p(x, θ - (π/2)·e_{l,i}) ] / 2
```

where `e_{l,i}` is the unit vector shifting only `θ_{l,i}`. (This is a standard identity for
Pauli-rotation gates — you may use it directly without re-deriving it.) By the chain rule:

```
∂Loss/∂θ_{l,i} = (2 / n_train) * Σ_i ( p(x_i,θ) - y_i ) * ∂p(x_i,θ)/∂θ_{l,i}
```

Starting from a given initial `θ_init`, run exactly `T` steps of **full-batch gradient
descent** with fixed learning rate `η`:

```
θ ← θ - η · ∇Loss(θ)
```

(Gradient computed fresh from the full training set at every step, using the parameter-shift
rule above for every one of the `L·n` parameters — that's `2·L·n` extra circuit simulations
per training step.) After `T` steps, use the final `θ` to predict on the query points.

---

### Input Format

```
n_train n L
x_1,1 x_1,2 ... x_1,n y_1
x_2,1 x_2,2 ... x_2,n y_2
...
x_n_train,1 ... x_n_train,n y_n_train
eta T
theta_0,0 theta_0,1 ... theta_0,n-1 theta_1,0 ... theta_L-1,n-1
m
q_1,1 q_1,2 ... q_1,n
...
q_m,1 q_m,2 ... q_m,n
```

- `theta_init` is given **flattened**, layer-major then qubit-index-minor (i.e. all of layer
  `0`'s `n` angles, then all of layer `1`'s, etc.) — exactly `L·n` numbers on one line.
- `eta > 0`, `T ≥ 0` (an integer number of gradient steps; `T = 0` means output predictions
  using `theta_init` unchanged).

### Output Format

Print `m` lines: the predicted `p(q, θ_final)` for each query point, with at least 6 digits
after the decimal point.

**Judging:** accepted if within `1e-4` absolute error of the reference solution (relative
tolerance is not used here, since all valid outputs lie in `[-1, 1]`).

---

### Constraints

- `1 ≤ n_train ≤ 50`
- `1 ≤ n ≤ 4` (qubits)
- `1 ≤ L ≤ 4` (variational layers)
- `1 ≤ m ≤ 50`
- `0 < eta ≤ 2`
- `0 ≤ T ≤ 20`
- All feature values, targets, and initial angles satisfy `|value| ≤ 10` (radians for angles),
  given with up to 6 decimal digits.
- Time limit: 4 seconds. Memory limit: 256 MB.

---

### Example 1

**Input**

```
2 2 1
0.5 -0.3 1.0
1.0 0.2 -1.0
0.5 3
0.1 -0.1
2
0.4 0.4
-0.2 0.6
```

**Output**

```
0.571864
0.794600
```

**Explanation:** `n_train=2` examples, `n=2` qubits, `L=1` layer, learning rate `0.5`, `T=3`
gradient descent steps from `θ_init = [0.1, -0.1]`. After simulating the encode→entangle→
rotate circuit, computing the parameter-shift gradient of the MSE loss over both training
points at every step, and applying `3` updates, the trained parameters converge to
approximately `θ ≈ [0.7482, -0.1000]`. Evaluating the trained circuit at the two query points
gives `0.571864` and `0.794600`.

---

### Example 2

**Input**

```
3 2 2
0.3 0.7 1.0
-0.5 0.1 -1.0
0.9 -0.4 0.5
0.3 2
0.2 0.0 -0.1 0.3
2
0.0 0.0
0.5 -0.5
```

**Output**

```
0.861363
0.967439
```

**Explanation:** Now `n=2` qubits but `L=2` layers (`4` trainable parameters total, given
flattened as `[θ_{0,0}, θ_{0,1}, θ_{1,0}, θ_{1,1}] = [0.2, 0.0, -0.1, 0.3]`), `3` training
examples, learning rate `0.3`, `T=2` steps. Each layer applies its own entangling chain before
its rotation — so the circuit here entangles the qubits **twice** before measurement. The
trained circuit evaluated at the two query points gives `0.861363` and `0.967439`.
