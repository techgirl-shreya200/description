# THE TORQUE MINIMIZER

Beginner | calculus | optimization

**Difficulty:** Easy
**Tags:** Calculus, Optimization

---

### Story

An Amazon warehouse robot arm's joint controller minimizes a quadratic torque-penalty cost function
before every movement, cheap enough to run at control-loop frequency, but it still needs a correct
gradient step, not an approximation.

---

### The Math

```
C(theta) = a * theta^2 + b * theta + c
dC/dtheta = 2*a*theta + b
```

Given `a, b, c` and a starting `theta_0`, take one gradient-descent step with learning rate `eta`:

```
theta_1 = theta_0 - eta * dC/dtheta at theta_0
```

### Input Format

```
a b c theta_0 eta
```

### Output Format

`theta_1` and `C(theta_1)`, space-separated, to 6 decimal places.

### Constraints

- `-100 <= a, b, c, theta_0 <= 100`, `a != 0`, `0 < eta <= 1`
- Time limit: 1.0 second.

---

### Example

**Input**

```
1.0 -4.0 5.0 0.0 0.1
```

**Output**

```
0.400000 3.560000
```

**Explanation:** `dC/dtheta` at `0` is `-4.0`, so `theta_1 = 0 - 0.1 * (-4.0) = 0.4`, and
`C(0.4) = 1(0.16) - 4(0.4) + 5 = 3.56`.
