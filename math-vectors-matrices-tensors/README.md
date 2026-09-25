# Vectors, matrices and tensors: shapes and basic operations

Beginner | linear-algebra | tensors

### The problem, from first principles

A single temperature reading is one number. A week of daily readings is a list of seven numbers. A month of readings across ten cities is a grid, ten rows, thirty-ish columns. A year of that, split by morning/afternoon/evening, stacks another axis on top of the grid. Nothing about the _idea_ changes as you add axes, only how many numbers you need to address one value: none, one, two, three.

That single idea, "a number, or numbers indexed by however many axes," is the entire foundation this curriculum builds on. Before touching gradients or neural networks, you need the vocabulary for describing the _shape_ of data and the two operations simple enough to define with no ambiguity at all: adding two same-shaped things, and multiplying them position by position.

### From theory to code

Theory names the ranks (scalar, vector, matrix, tensor) and the two attributes every one of them has: `shape` (how many entries along each axis) and `ndim` (how many axes there are). Implement four small functions that expose this vocabulary directly, plus the two operations that only ever touch matching positions.

Implement `shape_of`, `ndim_of`, `elementwise_add` and `elementwise_multiply` against that reasoning. The signatures and docstrings are already in the editor.

### Constraints

- `shape_of` returns a plain tuple, e.g. `(3,)` for a length-3 vector, `()` for a scalar.
- `ndim_of` returns a plain `int`.
- `elementwise_add` and `elementwise_multiply` assume `a` and `b` already share a shape (broadcasting is a later question's job).
- `elementwise_multiply` is NOT matrix multiplication: it must never reduce a vector down to a scalar.
- No Python loop over positions anywhere, this is meant to be vectorized.

### Hints

Open one at a time. Each gives away a little more than the last.

<details>
<summary>Hint 1</summary>

Every one of these already exists as a NumPy array attribute or operator. You're exposing vocabulary, not computing anything new.

</details>

<details>
<summary>Hint 2</summary>

`*` on two NumPy arrays is elementwise by default. Matrix multiplication is a separate operator (`@`), covered in a later question, don't reach for it here.

</details>
