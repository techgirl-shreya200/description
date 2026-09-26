# NETFLIX FAST-FORWARD

Advanced | transformers | linear-attention | linear-algebra | memory-optimization

**Difficulty:** Hard
**Tags:** Linear Algebra, Linear Attention, Transformers, Memory Optimization

---

### Story

Netflix's next-generation sequential recommendation engine uses a Transformer model to predict
exactly what a user will watch next, based on their entire historical timeline of clicks,
pauses, and completed shows. For power users, this sequence length `N` can easily reach
100,000 interactions.

Standard self-attention computes an `N x N` correlation matrix between all past interactions.
High-level deep learning libraries will blindly try to allocate a 100,000 x 100,000 matrix,
which is about 80 GB in float64, and crash an edge inference server with an out-of-memory
error.

You need to implement the mathematical foundation of **linear attention** from scratch. By
swapping softmax for a kernel feature map, you can exploit the associativity of matrix
multiplication and change the order of the multiplications so the `N x N` matrix never exists.
That takes the problem from an impossible `O(N^2 d)` to `O(N d^2)` time and `O(N d)` memory.

---

### The Math: Linear Attention

You are given three matrices describing the user's sequence: queries `Q`, keys `K`, and values
`V`. All three have shape `N x d`, where `N` is the sequence length and `d` is the embedding
dimension.

Standard attention computes the output `O` (shape `N x d`) as:

```
O = softmax(Q K^T) V
```

Materializing `Q K^T` costs `O(N^2)` memory and time.

To linearize, replace the exponential in softmax with a deterministic feature map `phi`
applied element-wise to `Q` and `K`. Here `phi` is a shifted ReLU, which keeps every value
strictly positive:

```
phi(x) = max(x, 0) + 1
```

Let `Q' = phi(Q)` and `K' = phi(K)`. Row `i` of the output is defined as:

```
O_i = ( sum_j (Q'_i . K'_j) V_j ) / ( sum_j (Q'_i . K'_j) )
```

Because `Q'_i . K'_j` is a scalar, `Q'_i` can be pulled out of both sums. Define:

```
M = (K')^T V        (a d x d matrix)
Z = sum_j K'_j      (a d-vector)
```

Then for every row at once, the numerator is `Q' M` and the denominator is `Q' Z`:

```
O_i = (Q' M)_i / (Q' Z)_i
```

Computing `M` and `Z` first keeps peak memory at `O(N d)` and time at `O(N d^2)`.

---

### Input Format

```
N d
q_{1,1} q_{1,2} ... q_{1,d}
...                              (N lines for Q)
k_{1,1} k_{1,2} ... k_{1,d}
...                              (N lines for K)
v_{1,1} v_{1,2} ... v_{1,d}
...                              (N lines for V)
```

- `N` is the sequence length and `d` is the embedding dimension.
- The next `N` lines are the rows of `Q`, then `N` lines for `K`, then `N` lines for `V`.
  Every element is a float.

### Output Format

Do not print the full `N x d` output matrix. Print `N` lines. Line `i` is the **sum of the
elements** of row `i` of `O`, to exactly 6 decimal places:

```
sum_{c=1..d} O_{i,c}
```

**Judging:** accepted if within `1e-5` absolute error of the reference solution.

---

### Constraints

- `1 <= N <= 10^5`
- `1 <= d <= 64`
- Every element of `Q`, `K`, and `V` satisfies `-10 <= x <= 10`
- Time limit: 0.5 seconds.
- Memory limit: 64 MB. A dense `N x N` float matrix at `N = 10^5` needs about 80 GB, so
  building one fails immediately.

---

### Example 1

**Input**

```
3 2
1.0 -2.0
0.0 1.0
-1.0 0.5
2.0 -1.0
0.5 0.5
-0.5 1.0
1.0 2.0
-1.0 0.5
0.0 -1.0
```

**Output**

```
0.951613
0.534483
0.622449
```

**Explanation:** Apply `phi(x) = max(x, 0) + 1` to `Q` and `K`:

```
Q' = [[2.0, 1.0],      K' = [[3.0, 1.0],
      [1.0, 2.0],            [1.5, 1.5],
      [1.0, 1.5]]            [1.0, 2.0]]
```

Then `M = (K')^T V = [[1.5, 5.75], [-0.5, 0.75]]` and `Z = [5.5, 4.5]`. The numerator
`Q' M` is `[[2.5, 12.25], [0.5, 7.25], [0.75, 6.875]]` and the denominator `Q' Z` is
`[15.5, 14.5, 12.25]`. Row 1 sums to `(2.5 + 12.25) / 15.5 = 0.951613`, row 2 to
`(0.5 + 7.25) / 14.5 = 0.534483`, and row 3 to `(0.75 + 6.875) / 12.25 = 0.622449`.

---

### Example 2

**Input**

```
2 2
-1.0 -2.0
-3.0 -1.0
-1.0 -1.0
-2.0 -3.0
1.0 2.0
3.0 4.0
```

**Output**

```
5.000000
5.000000
```

**Explanation:** Every element of `Q` and `K` is negative, so `max(x, 0)` zeroes them all and
`phi` returns exactly `1` everywhere. Every query then attends to every key with equal weight,
so each output row is the plain average of the value rows, `[2.0, 3.0]`, which sums to `5.0`.
This is the case the `+1` shift exists for: without it the denominator would be `0`.
