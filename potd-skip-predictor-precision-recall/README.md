# THE SKIP PREDICTOR

Beginner | metrics-and-evaluation | classification

**Difficulty:** Easy
**Tags:** Metrics & Evaluation, Classification

---

### Story

Spotify's skip-predictor flags tracks a listener is likely to abandon within 5 seconds. Before the
model is allowed near production traffic, it has to clear a basic precision/recall sanity check on
a held-out batch.

---

### The Math

```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * Precision * Recall / (Precision + Recall)
```

### Input Format

```
n
p_1 y_1
p_2 y_2
...
p_n y_n
```

`p_i` is the predicted binary label, `y_i` the true label, both `0` or `1`.

### Output Format

`precision recall f1`, space-separated, to 6 decimals.

### Constraints

- `1 <= n <= 10^6`
- Time limit: 1.0 second.

---

### Example

**Input**

```
8
1 1
1 0
0 0
1 1
0 0
0 1
1 1
0 0
```

**Output**

```
0.750000 0.750000 0.750000
```

**Explanation:** `TP = 3` (rows 1, 4, 7), `FP = 1` (row 2), `FN = 1` (row 6).
`P = 3/4 = 0.75`, `R = 3/4 = 0.75`, `F1 = 0.75`.
