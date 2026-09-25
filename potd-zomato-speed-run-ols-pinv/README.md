# ZOMATO SPEED-RUN

Beginner | classical-ml | linear-algebra

**Difficulty:** Easy
**Tags:** Classic ML, Linear Algebra

---

### Story

In Zomato's early growth phase, the delivery-time feature had zero pipeline maturity: no
gradient-boosted trees, no feature store, just a spreadsheet of past deliveries. The team needs a
linear estimator working today. You are given distance and historical delivery time for `n` past
orders; fit ordinary least squares via the closed-form normal equation and predict on new orders.

---

### The Math

With the intercept folded into `X` as a leading column of ones:

```
w = (X^T X)^-1 X^T y
```

Because production data occasionally has duplicated or near-collinear feature columns, a literal
inverse can fail. Use the Moore-Penrose pseudo-inverse (SVD-based) instead of a hard inverse.

### Input Format

```
n d
x_1,1 ... x_1,d y_1
...
x_n,1 ... x_n,d y_n
m
q_1,1 ... q_1,d
...
q_m,1 ... q_m,d
```

### Output Format

First line: fitted coefficients (intercept, then `w_1..w_d`) to 6 decimals. Next `m` lines:
predicted values to 6 decimals.

### Constraints

- `1 <= n <= 10^4`, `1 <= d <= 20`, `1 <= m <= 10^3`
- Time limit: 2.0 seconds. Memory: 256 MB.

---

### Example

**Input**

```
3 1
1 5
2 7
3 9
2
4
5
```

**Output**

```
3.000000 2.000000
11.000000
13.000000
```

**Explanation:** The data is exactly `y = 2x + 3`, so the fit recovers it exactly: intercept `3.0`,
slope `2.0`. Predictions at `x = 4, 5` are `11` and `13`.
