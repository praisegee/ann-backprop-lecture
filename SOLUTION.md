# Backpropagation exercise: worked solution

Network: 3 inputs (units 1, 2, 3), hidden units 4 and 5, output unit 6, sigmoid activation.
One training example, updated after that example (case updating).

## Given

| Parameter | Value | Parameter | Value |
|---|---|---|---|
| w14 | 0.2 | w15 | -0.3 |
| w24 | 0.4 | w25 | 0.1 |
| w34 | -0.5 | w35 | 0.2 |
| w46 | -0.3 | w56 | -0.2 |
| b4 | -0.4 | b5 | 0.2 |
| b6 | 0.1 | l (learning rate) | 0.9 |
| x1, x2, x3 | 1, 0, 1 | T (target) | 1 |

## Formulas (from the notes)

- Net input: `I_j = Σ w_ij·O_i + b_j`
- Sigmoid output: `O_j = 1 / (1 + e^(−I_j))`
- Output-unit error: `Err_k = O_k(1 − O_k)(T_k − O_k)`
- Hidden-unit error: `Err_j = O_j(1 − O_j) Σ Err_k·w_jk`
- Weight update: `Δw_ij = l·Err_j·O_i`, then `w_ij ← w_ij + Δw_ij`
- Bias update: `Δb_j = l·Err_j`, then `b_j ← b_j + Δb_j`

(For input units, O_i = x_i.)

---

## Step 1: Forward propagation

### Hidden unit 4
```
I4 = w14·x1 + w24·x2 + w34·x3 + b4
   = (0.2)(1) + (0.4)(0) + (−0.5)(1) + (−0.4)
   = 0.2 + 0 − 0.5 − 0.4
   = −0.7
O4 = 1 / (1 + e^0.7) = 1 / (1 + 2.0138) = 0.3318
```

### Hidden unit 5
```
I5 = w15·x1 + w25·x2 + w35·x3 + b5
   = (−0.3)(1) + (0.1)(0) + (0.2)(1) + 0.2
   = −0.3 + 0 + 0.2 + 0.2
   = 0.1
O5 = 1 / (1 + e^−0.1) = 1 / (1 + 0.9048) = 0.5250
```

### Output unit 6
```
I6 = w46·O4 + w56·O5 + b6
   = (−0.3)(0.3318) + (−0.2)(0.5250) + 0.1
   = −0.0995 − 0.1050 + 0.1
   = −0.1045   (≈ −0.105)
O6 = 1 / (1 + e^0.1045) = 1 / (1 + 1.1102) = 0.4739
```

| Unit j | Net input I_j | Output O_j |
|---|---|---|
| 4 | −0.7000 | 0.3318 |
| 5 | 0.1000 | 0.5250 |
| 6 | −0.1045 | **0.4739** |

The network predicts 0.4739 but the target is 1, so the error is T − O6 = 0.5261.

---

## Step 2: Backward propagation (errors)

### Output unit 6
```
Err6 = O6(1 − O6)(T − O6)
     = (0.4739)(1 − 0.4739)(1 − 0.4739)
     = (0.4739)(0.5261)(0.5261)
     = 0.1312
```

### Hidden unit 5 (it feeds only unit 6)
```
Err5 = O5(1 − O5)·Err6·w56
     = (0.5250)(1 − 0.5250)(0.1312)(−0.2)
     = (0.2494)(−0.02624)
     = −0.0065
```

### Hidden unit 4 (it feeds only unit 6)
```
Err4 = O4(1 − O4)·Err6·w46
     = (0.3318)(1 − 0.3318)(0.1312)(−0.3)
     = (0.2217)(−0.03936)
     = −0.0087
```

| Unit j | Err_j |
|---|---|
| 6 | 0.1312 |
| 5 | −0.0065 |
| 4 | −0.0087 |

> Hidden errors are computed with the **old** weights w46 and w56, before any weight is updated.

---

## Step 3: Weight updates, `w_new = w + l·Err_j·O_i`

| Weight | Old | Δw = l·Err_j·O_i | New |
|---|---|---|---|
| w46 | −0.3 | (0.9)(0.1312)(0.3318) = +0.0392 | **−0.2608** |
| w56 | −0.2 | (0.9)(0.1312)(0.5250) = +0.0620 | **−0.1380** |
| w14 | 0.2 | (0.9)(−0.0087)(1) = −0.0079 | **0.1921** |
| w15 | −0.3 | (0.9)(−0.0065)(1) = −0.0059 | **−0.3059** |
| w24 | 0.4 | (0.9)(−0.0087)(0) = 0 | **0.4000** |
| w25 | 0.1 | (0.9)(−0.0065)(0) = 0 | **0.1000** |
| w34 | −0.5 | (0.9)(−0.0087)(1) = −0.0079 | **−0.5079** |
| w35 | 0.2 | (0.9)(−0.0065)(1) = −0.0059 | **0.1941** |

w24 and w25 do not change because x2 = 0.

## Step 4: Bias updates, `b_new = b + l·Err_j`

| Bias | Old | Δb = l·Err_j | New |
|---|---|---|---|
| b6 | 0.1 | (0.9)(0.1312) = +0.1181 | **0.2181** |
| b5 | 0.2 | (0.9)(−0.0065) = −0.0059 | **0.1941** |
| b4 | −0.4 | (0.9)(−0.0087) = −0.0079 | **−0.4079** |

---

## Check: forward pass again with the updated parameters

| | Output O6 | Error T − O6 |
|---|---|---|
| Before update | 0.4739 | 0.5261 |
| After one update | 0.5152 | 0.4848 |

The output moved toward the target, so the update worked. Repeating these steps over more
epochs pushes O6 closer to 1, until a stopping condition is met: an error threshold, a fixed
number of epochs, or the misclassification rate.

Run `python3 01_backprop_exercise.py` to reproduce every number above, or `lectrace serve` to step through it interactively.
