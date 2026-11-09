"""
Worked solution: one forward pass + one backward pass (case updating)
for the 3-2-1 sigmoid network from the lecture notes.

Network:
    inputs x1, x2, x3  -> units 1, 2, 3
    hidden units 4, 5   (biases b4, b5)
    output unit 6       (bias b6)

Formulas used (from the notes):
    Net input        I_j   = sum_i w_ij * O_i + b_j
    Output (sigmoid) O_j   = 1 / (1 + e^(-I_j))
    Output error     Err_k = O_k (1 - O_k) (T_k - O_k)
    Hidden error     Err_j = O_j (1 - O_j) * sum_k Err_k * w_jk
    Weight update    dw_ij = l * Err_j * O_i ;  w_ij <- w_ij + dw_ij
    Bias update      db_j  = l * Err_j       ;  b_j  <- b_j  + db_j
"""
import math

# ---- Given values ---------------------------------------------------------
x = {1: 1, 2: 0, 3: 1}
w = {
    (1, 4): 0.2, (1, 5): -0.3,
    (2, 4): 0.4, (2, 5): 0.1,
    (3, 4): -0.5, (3, 5): 0.2,
    (4, 6): -0.3, (5, 6): -0.2,
}
b = {4: -0.4, 5: 0.2, 6: 0.1}
l = 0.9   # learning rate
T = 1     # target


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


O = dict(x)  # outputs of input units are just the inputs
I = {}

print("=" * 64)
print("FORWARD PROPAGATION")
print("=" * 64)
for j in (4, 5, 6):
    srcs = [i for (i, k) in w if k == j]
    terms = " + ".join(f"({w[(i, j)]})({O[i]:.4f})" for i in srcs)
    I[j] = sum(w[(i, j)] * O[i] for i in srcs) + b[j]
    O[j] = sigmoid(I[j])
    print(f"I{j} = {terms} + ({b[j]}) = {I[j]:.4f}")
    print(f"O{j} = 1 / (1 + e^-({I[j]:.4f})) = {O[j]:.4f}\n")

print("=" * 64)
print("BACKWARD PROPAGATION (errors)")
print("=" * 64)
Err = {}
Err[6] = O[6] * (1 - O[6]) * (T - O[6])
print(f"Err6 = O6(1-O6)(T-O6) = ({O[6]:.4f})(1-{O[6]:.4f})({T}-{O[6]:.4f})"
      f" = {Err[6]:.4f}")
for j in (5, 4):
    Err[j] = O[j] * (1 - O[j]) * Err[6] * w[(j, 6)]
    print(f"Err{j} = O{j}(1-O{j}) Err6 w{j}6 = ({O[j]:.4f})(1-{O[j]:.4f})"
          f"({Err[6]:.4f})({w[(j, 6)]}) = {Err[j]:.4f}")

print()
print("=" * 64)
print("WEIGHT UPDATES   w_new = w + l * Err_j * O_i")
print("=" * 64)
new_w = {}
for (i, j) in [(4, 6), (5, 6), (1, 4), (1, 5), (2, 4), (2, 5), (3, 4), (3, 5)]:
    dw = l * Err[j] * O[i]
    new_w[(i, j)] = w[(i, j)] + dw
    print(f"w{i}{j} = {w[(i, j)]:>5} + ({l})({Err[j]:.4f})({O[i]:.4f})"
          f" = {w[(i, j)]:>5} + ({dw:.4f}) = {new_w[(i, j)]:.4f}")

print()
print("=" * 64)
print("BIAS UPDATES   b_new = b + l * Err_j")
print("=" * 64)
new_b = {}
for j in (6, 5, 4):
    db = l * Err[j]
    new_b[j] = b[j] + db
    print(f"b{j} = {b[j]:>5} + ({l})({Err[j]:.4f}) = {b[j]:>5} + ({db:.4f})"
          f" = {new_b[j]:.4f}")

# ---- Check: forward pass with updated parameters --------------------------
O2 = dict(x)
for j in (4, 5, 6):
    srcs = [i for (i, k) in new_w if k == j]
    O2[j] = sigmoid(sum(new_w[(i, j)] * O2[i] for i in srcs) + new_b[j])
print()
print("=" * 64)
print(f"Output before update: {O[6]:.4f}   error (T-O) = {T - O[6]:.4f}")
print(f"Output after  update: {O2[6]:.4f}   error (T-O) = {T - O2[6]:.4f}")
