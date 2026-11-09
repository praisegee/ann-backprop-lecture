from lectrace import text, note, table, image
import math


def main():
    text("# Backpropagation Exercise 2: A 2-2-1 Network")
    text(
        "A second worked example using the same method as Lecture 1, on a smaller network. "
        "The two questions are:\n\n"
        "1. **Compute the final output** (forward propagation).\n"
        "2. **Update the weights and the biases** (backward propagation)."
    )

    # @state The network
    text("## 1. The network")
    text("Two inputs (units 1, 2), two hidden units (3, 4), one output unit (5), all with sigmoid activation:")
    image("network2.svg", width=560)
    text("Each input unit connects to both hidden units, and both hidden units connect to the output unit.")

    # @state Given values
    text("## 2. Given values")
    x = {1: 1, 2: 1}  # @inspect x
    w = {  # @inspect w
        "w13": 0.5, "w14": 0.9,
        "w23": 0.4, "w24": 1.0,
        "w35": -1.2, "w45": 1.1,
    }
    b = {"b3": -0.8, "b4": 0.1, "b5": -0.3}  # @inspect b
    l = 0.5  # @inspect l
    T = 0.9  # @inspect T
    text("Inputs $x = (1, 1)$, target $T = 0.9$, learning rate $l = 0.5$.")
    table(
        [
            ["w13", 0.5, "w14", 0.9],
            ["w23", 0.4, "w24", 1.0],
            ["w35", -1.2, "w45", 1.1],
            ["b3", -0.8, "b4", 0.1],
            ["b5", -0.3, "l", 0.5],
        ],
        headers=["Parameter", "Value", "Parameter", "Value"],
        caption="Initial weights and biases",
    )

    # @state Formulas
    text("## 3. The formulas (same as Lecture 1)")
    text("$$I_j = \\sum_i w_{ij}\\,O_i + b_j \\qquad O_j = \\frac{1}{1 + e^{-I_j}}$$")
    text("$$Err_k = O_k(1 - O_k)(T_k - O_k) \\qquad Err_j = O_j(1 - O_j)\\sum_k Err_k\\, w_{jk}$$")
    text("$$\\Delta w_{ij} = l\\cdot Err_j\\cdot O_i \\qquad \\Delta b_j = l\\cdot Err_j$$")
    note("Values are rounded to 3 decimal places at each step, as you would on paper.")

    # @state Forward pass
    text("## 4. Question 1: Forward propagation")
    text("Go through the layers left to right. Step into `net_input` and `sigmoid` to see each sum being built.")

    text("### Hidden unit 3")
    I3 = net_input([x[1], x[2]], [w["w13"], w["w23"]], b["b3"])  # @inspect I3
    O3 = sigmoid(I3)  # @inspect O3
    text(
        f"$I_3 = w_{{13}}x_1 + w_{{23}}x_2 + b_3 = (0.5)(1) + (0.4)(1) + (-0.8) = {I3:.3f}$\n\n"
        f"$O_3 = \\dfrac{{1}}{{1 + e^{{{-I3:.3f}}}}} = {O3:.3f}$"
    )

    text("### Hidden unit 4")
    I4 = net_input([x[1], x[2]], [w["w14"], w["w24"]], b["b4"])  # @inspect I4
    O4 = sigmoid(I4)  # @inspect O4
    text(
        f"$I_4 = w_{{14}}x_1 + w_{{24}}x_2 + b_4 = (0.9)(1) + (1.0)(1) + 0.1 = {I4:.3f}$\n\n"
        f"$O_4 = \\dfrac{{1}}{{1 + e^{{{-I4:.3f}}}}} = {O4:.3f}$"
    )

    text("### Output unit 5")
    text("The inputs to unit 5 are the **outputs of the hidden units**, $O_3$ and $O_4$.")
    I5 = net_input([O3, O4], [w["w35"], w["w45"]], b["b5"])  # @inspect I5
    O5 = sigmoid(I5)  # @inspect O5
    text(
        f"$I_5 = w_{{35}}O_3 + w_{{45}}O_4 + b_5 = (-1.2)({O3:.3f}) + (1.1)({O4:.3f}) + (-0.3) = {I5:.3f}$\n\n"
        f"$O_5 = \\dfrac{{1}}{{1 + e^{{{-I5:.3f}}}}} = {O5:.3f}$"
    )
    table(
        [[3, I3, O3], [4, I4, O4], [5, I5, O5]],
        headers=["Unit j", "Net input I_j", "Output O_j"],
        caption="Forward pass results",
    )
    text(
        f"**Answer to Question 1: the final output is $O_5 = {O5:.3f}$.** "
        f"The target is $0.9$, so the network is off by $T - O_5 = {T - O5:.3f}$."
    )

    # @state Backward pass
    text("## 5. Question 2, part A: Backward propagation of the errors")
    text("Go right to left. Start at the output unit, then pass its error back to the hidden units.")

    text("### Output unit 5")
    error = round(T - O5, 3)  # @inspect error
    text(f"Raw output error: $T - O_5 = 0.9 - {O5:.3f} = {error:.3f}$")
    text(f"Sigmoid derivative: $O_5(1-O_5) = ({O5:.3f})({1 - O5:.3f}) = {O5 * (1 - O5):.3f}$")
    Err5 = output_error(O5, T)  # @inspect Err5
    text(
        f"$$Err_5 = O_5(1-O_5)(T-O_5) = ({O5:.3f})({1 - O5:.3f})({error:.3f}) = \\mathbf{{{Err5:.3f}}}$$"
    )
    text("$Err_5$ is positive, so the output must **go up** towards 0.9.")

    text("### Hidden unit 3")
    text("Unit 3 feeds only unit 5, so the sum has a single term: $Err_5\\,w_{35}$.")
    text(
        f"Sigmoid derivative: $O_3(1-O_3) = ({O3:.3f})({1 - O3:.3f}) = {O3 * (1 - O3):.3f}$  \n"
        f"Error sent back from unit 5: $Err_5\\,w_{{35}} = ({Err5:.3f})(-1.2) = {Err5 * w['w35']:.4f}$"
    )
    Err3 = hidden_error(O3, [Err5], [w["w35"]])  # @inspect Err3
    text(
        f"$$Err_3 = O_3(1-O_3)\\,Err_5\\,w_{{35}} = ({O3:.3f})({1 - O3:.3f})({Err5:.3f})(-1.2) = \\mathbf{{{Err3:.3f}}}$$"
    )

    text("### Hidden unit 4")
    text(
        f"Sigmoid derivative: $O_4(1-O_4) = ({O4:.3f})({1 - O4:.3f}) = {O4 * (1 - O4):.3f}$  \n"
        f"Error sent back from unit 5: $Err_5\\,w_{{45}} = ({Err5:.3f})(1.1) = {Err5 * w['w45']:.4f}$"
    )
    Err4 = hidden_error(O4, [Err5], [w["w45"]])  # @inspect Err4
    text(
        f"$$Err_4 = O_4(1-O_4)\\,Err_5\\,w_{{45}} = ({O4:.3f})({1 - O4:.3f})({Err5:.3f})(1.1) = \\mathbf{{{Err4:.3f}}}$$"
    )
    image("backprop2.svg", width=560)
    note(
        "Err3 is negative because w35 is negative: unit 3 pushes the output down, "
        "so it should become less active. Err4 is positive because w45 is positive."
    )
    note(
        f"Err4 is small even though w45 is large: O4 = {O4:.3f} is close to 1, where the "
        f"sigmoid is flat, so its derivative O4(1 − O4) = {O4 * (1 - O4):.3f} is small."
    )
    table(
        [
            [5, O5, "O5(1−O5)(T−O5)", f"({O5:.3f})({1 - O5:.3f})({error:.3f})", Err5],
            [3, O3, "O3(1−O3)·Err5·w35", f"({O3:.3f})({1 - O3:.3f})({Err5:.3f})(−1.2)", Err3],
            [4, O4, "O4(1−O4)·Err5·w45", f"({O4:.3f})({1 - O4:.3f})({Err5:.3f})(1.1)", Err4],
        ],
        headers=["Unit j", "O_j", "Formula", "Substitution", "Err_j"],
        caption="Error values from backpropagation",
    )
    note("The hidden errors use the OLD weights w35 and w45. Compute every error first, then update.")

    # @state Weight updates
    text("## 6. Question 2, part B: Updating the weights")
    text("For every connection $i \\to j$: $\\;\\Delta w_{ij} = l \\cdot Err_j \\cdot O_i$.")
    O = {1: x[1], 2: x[2], 3: O3, 4: O4}
    Err = {3: Err3, 4: Err4, 5: Err5}
    new_w = {}  # @inspect new_w
    rows = []
    for name in ["w35", "w45", "w13", "w23", "w14", "w24"]:
        i, j = int(name[1]), int(name[2])
        new_w[name], dw = update_weight(w[name], l, Err[j], O[i])  # @inspect new_w
        rows.append([name, w[name], f"Err{j} = {Err[j]:.3f}", f"O{i} = {O[i]:.3f}",
                     f"({l})({Err[j]:.3f})({O[i]:.3f}) = {dw:+.3f}", new_w[name]])
    table(rows, headers=["Weight", "Old", "Err_j", "O_i", "Δw = l·Err_j·O_i", "New"], caption="Weight updates")

    # @state Bias updates
    text("## 7. Question 2, part C: Updating the biases")
    text("For every unit $j$: $\\;\\Delta b_j = l \\cdot Err_j$.")
    new_b = {}  # @inspect new_b
    rows = []
    for name in ["b5", "b3", "b4"]:
        j = int(name[1])
        new_b[name], db = update_bias(b[name], l, Err[j])  # @inspect new_b
        rows.append([name, b[name], f"Err{j} = {Err[j]:.3f}", f"({l})({Err[j]:.3f}) = {db:+.3f}", new_b[name]])
    table(rows, headers=["Bias", "Old", "Err_j", "Δb = l·Err_j", "New"], caption="Bias updates")

    # @state Summary
    text("## 8. Summary and check")
    table(
        [[k, w[k], new_w[k]] for k in ["w13", "w14", "w23", "w24", "w35", "w45"]]
        + [[k, b[k], new_b[k]] for k in ["b3", "b4", "b5"]],
        headers=["Parameter", "Before", "After"],
        caption="All parameters after one update",
    )
    text("Run the forward pass again with the updated parameters:")
    O3n = sigmoid(net_input([x[1], x[2]], [new_w["w13"], new_w["w23"]], new_b["b3"]))  # @stepover
    O4n = sigmoid(net_input([x[1], x[2]], [new_w["w14"], new_w["w24"]], new_b["b4"]))  # @stepover
    O5n = sigmoid(net_input([O3n, O4n], [new_w["w35"], new_w["w45"]], new_b["b5"]))  # @stepover @inspect O5n
    table(
        [["Before update", O5, round(T - O5, 3)], ["After one update", O5n, round(T - O5n, 3)]],
        headers=["", "Output O5", "Error T − O5"],
    )
    text(
        f"The output rose from **{O5:.3f}** to **{O5n:.3f}**, closer to the target of 0.9. "
        "Repeating this over more epochs keeps shrinking the error."
    )


def net_input(inputs, weights, bias):
    total = bias
    for o, wt in zip(inputs, weights):
        total += wt * o
    return round(total, 3)


def sigmoid(z):
    return round(1 / (1 + math.exp(-z)), 3)


def output_error(o, target):
    derivative = o * (1 - o)
    error = target - o
    return round(derivative * error, 3)


def hidden_error(o, next_errors, next_weights):
    derivative = o * (1 - o)
    back = sum(e * wt for e, wt in zip(next_errors, next_weights))
    return round(derivative * back, 3)


def update_weight(old, lr, err, o_in):
    delta = round(lr * err * o_in, 3) + 0.0  # + 0.0 turns -0.0 into 0.0
    return round(old + delta, 3), delta


def update_bias(old, lr, err):
    delta = round(lr * err, 3) + 0.0
    return round(old + delta, 3), delta


if __name__ == "__main__":
    main()
