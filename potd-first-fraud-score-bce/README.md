# THE FIRST FRAUD SCORE

Beginner | loss-functions

**Difficulty:** Easy
**Tags:** Loss Functions

---

### Story

PayPal's very first fraud model outputs a raw probability per transaction. Before anyone trusts the
model, its loss on a labeled calibration batch needs to be computed exactly, matching the framework
reference to the last digit.

---

### The Math

```
L = -(1/n) * sum_i [ y_i * log(yhat_i) + (1 - y_i) * log(1 - yhat_i) ]
```

### Input Format

```
n
y_1 yhat_1
...
y_n yhat_n
```

### Output Format

Scalar loss, 6 decimals.

### Constraints

- `1 <= n <= 10^5`, `10^-7 <= yhat_i <= 1 - 10^-7`
- Time limit: 1.0 second.

---

### Example

**Input**

```
4
1 0.9
0 0.2
1 0.6
0 0.3
```

**Output**

```
0.299001
```
