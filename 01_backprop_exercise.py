from lectrace import text, note, table, image
import math


def main():
    text("# Training an ANN with Gradient Descent: A Worked Example")
    text(
        "We take one training example through a small neural network: a **forward pass** "
        "to get a prediction, then a **backward pass** to push the error back and "
        "update every weight and bias."
    )
    text(
        "Gradient descent means taking small steps that lower the error, so that the new "
        "parameters $\\theta^*$ satisfy $f(\\theta^*) < f(\\theta)$. Each update has the form "
        "$\\theta_{i+1} = \\theta_i + \\Delta\\theta_i$."
    )

    # @state The network
    text("## 1. The network")
    text("Three inputs (units 1, 2, 3), two hidden units (4, 5), one output unit (6), all with sigmoid activation:")
    image("network.svg", width=620)
    text("Every input unit connects to both hidden units, and both hidden units connect to the output unit.")

    # @state Given values
    text("## 2. Given values")
    x = {1: 1, 2: 0, 3: 1}  # @inspect x
    w = {  # @inspect w
        "w14": 0.2, "w15": -0.3,
        "w24": 0.4, "w25": 0.1,
        "w34": -0.5, "w35": 0.2,
        "w46": -0.3, "w56": -0.2,
    }
    b = {"b4": -0.4, "b5": 0.2, "b6": 0.1}  # @inspect b
    l = 0.9  # @inspect l
    T = 1  # @inspect T
    text("Inputs $x = (1, 0, 1)$, target $T = 1$, learning rate $l = 0.9$.")
    table(
        [
            ["w14", 0.2, "w15", -0.3],
            ["w24", 0.4, "w25", 0.1],
            ["w34", -0.5, "w35", 0.2],
            ["w46", -0.3, "w56", -0.2],
            ["b4", -0.4, "b5", 0.2],
            ["b6", 0.1, "l", 0.9],
        ],
        headers=["Parameter", "Value", "Parameter", "Value"],
        caption="Initial weights and biases",
    )

    # @state Formulas
    text("## 3. The formulas")
    text("**Net input** to unit $j$ (a weighted sum of the outputs that feed it, plus its bias):")
    text("$$I_j = \\sum_i w_{ij}\\,O_i + b_j$$")
    text("**Output** of unit $j$ (the sigmoid activation):")
    text("$$O_j = \\frac{1}{1 + e^{-I_j}}$$")
    text("**Error at an output unit** $k$:")
    text("$$Err_k = O_k(1 - O_k)(T_k - O_k)$$")
    text("**Error at a hidden unit** $j$ (the errors of the units it feeds, sent back through the weights):")
    text("$$Err_j = O_j(1 - O_j)\\sum_k Err_k\\, w_{jk}$$")
    text("**Updates**:")
    text("$$\\Delta w_{ij} = l\\cdot Err_j\\cdot O_i, \\qquad w_{ij} \\leftarrow w_{ij} + \\Delta w_{ij}$$")
    text("$$\\Delta b_j = l\\cdot Err_j, \\qquad b_j \\leftarrow b_j + \\Delta b_j$$")
    note("O(1 − O) is the derivative of the sigmoid, which is why it appears in both error formulas.")
    note("For an input unit, the output is just the input: Oᵢ = xᵢ.")

    # @state Forward pass
    text("## 4. Forward propagation")
    text("Go through the layers left to right. Step into `net_input` and `sigmoid` to see each sum being built.")

    text("### Hidden unit 4")
    I4 = net_input([x[1], x[2], x[3]], [w["w14"], w["w24"], w["w34"]], b["b4"])  # @inspect I4
    O4 = sigmoid(I4)  # @inspect O4
    text(
        f"$I_4 = (0.2)(1) + (0.4)(0) + (-0.5)(1) + (-0.4) = {I4:.3f}$\n\n"
        f"$O_4 = \\dfrac{{1}}{{1 + e^{{0.7}}}} = {O4:.3f}$"
    )

    text("### Hidden unit 5")
    I5 = net_input([x[1], x[2], x[3]], [w["w15"], w["w25"], w["w35"]], b["b5"])  # @inspect I5
    O5 = sigmoid(I5)  # @inspect O5
    text(
        f"$I_5 = (-0.3)(1) + (0.1)(0) + (0.2)(1) + 0.2 = {I5:.3f}$\n\n"
        f"$O_5 = \\dfrac{{1}}{{1 + e^{{-0.1}}}} = {O5:.3f}$"
    )

    text("### Output unit 6")
    text("The inputs to unit 6 are the **outputs of the hidden units**, $O_4$ and $O_5$.")
    I6 = net_input([O4, O5], [w["w46"], w["w56"]], b["b6"])  # @inspect I6
    O6 = sigmoid(I6)  # @inspect O6
    text(
        f"$I_6 = (-0.3)({O4:.3f}) + (-0.2)({O5:.3f}) + 0.1 = {I6:.3f}$\n\n"
        f"$O_6 = \\dfrac{{1}}{{1 + e^{{{-I6:.3f}}}}} = {O6:.3f}$"
    )

    table(
        [[4, round(I4, 3), round(O4, 3)], [5, round(I5, 3), round(O5, 3)], [6, round(I6, 3), round(O6, 3)]],
        headers=["Unit j", "Net input I_j", "Output O_j"],
        caption="Forward pass results",
    )
    text(
        f"The network predicts **{O6:.3f}**, but the target is **1**. "
        f"It is off by $T - O_6 = {T - O6:.3f}$, so we need to correct the parameters."
    )

    # @state Backward pass
    text("## 5. Backward propagation: the errors")
    text("Go right to left. Start at the output unit, then pass its error back to the hidden units.")

    text("### Output unit 6")
    error = round(T - O6, 3)  # @inspect error
    text(f"Raw output error: $T - O_6 = 1 - {O6:.3f} = {error:.3f}$")
    text(f"Sigmoid derivative: $O_6(1-O_6) = ({O6:.3f})({1 - O6:.3f}) = {O6 * (1 - O6):.3f}$")
    Err6 = output_error(O6, T)  # @inspect Err6
    text(
        f"$$Err_6 = O_6(1-O_6)(T-O_6) = ({O6:.3f})({1 - O6:.3f})({error:.3f}) = \\mathbf{{{Err6:.3f}}}$$"
    )
    text("$Err_6$ is positive, so the output must **go up**.")

    text("### Hidden unit 5")
    text("Unit 5 feeds only unit 6, so the sum $\\sum_k Err_k w_{5k}$ has a single term: $Err_6\\,w_{56}$.")
    text(
        f"Sigmoid derivative: $O_5(1-O_5) = ({O5:.3f})({1 - O5:.3f}) = {O5 * (1 - O5):.3f}$  \n"
        f"Error sent back from unit 6: $Err_6\\,w_{{56}} = ({Err6:.3f})(-0.2) = {Err6 * w['w56']:.4f}$"
    )
    Err5 = hidden_error(O5, [Err6], [w["w56"]])  # @inspect Err5
    text(
        f"$$Err_5 = O_5(1-O_5)\\,Err_6\\,w_{{56}} = ({O5:.3f})({1 - O5:.3f})({Err6:.3f})(-0.2) = \\mathbf{{{Err5:.3f}}}$$"
    )

    text("### Hidden unit 4")
    text(
        f"Sigmoid derivative: $O_4(1-O_4) = ({O4:.3f})({1 - O4:.3f}) = {O4 * (1 - O4):.3f}$  \n"
        f"Error sent back from unit 6: $Err_6\\,w_{{46}} = ({Err6:.3f})(-0.3) = {Err6 * w['w46']:.4f}$"
    )
    Err4 = hidden_error(O4, [Err6], [w["w46"]])  # @inspect Err4
    text(
        f"$$Err_4 = O_4(1-O_4)\\,Err_6\\,w_{{46}} = ({O4:.3f})({1 - O4:.3f})({Err6:.3f})(-0.3) = \\mathbf{{{Err4:.3f}}}$$"
    )
    image("backprop.svg", width=620)
    note(
        "The hidden errors use the OLD weights w46 and w56. "
        "Compute every error first, then update."
    )
    note(
        "The hidden errors are negative because w46 and w56 are negative: "
        "a hidden unit that is too active pushes the output down."
    )
    table(
        [
            [6, O6, "O6(1−O6)(T−O6)", f"({O6:.3f})({1 - O6:.3f})({error:.3f})", Err6],
            [5, O5, "O5(1−O5)·Err6·w56", f"({O5:.3f})({1 - O5:.3f})({Err6:.3f})(−0.2)", Err5],
            [4, O4, "O4(1−O4)·Err6·w46", f"({O4:.3f})({1 - O4:.3f})({Err6:.3f})(−0.3)", Err4],
        ],
        headers=["Unit j", "O_j", "Formula", "Substitution", "Err_j"],
        caption="Error values from backpropagation",
    )
    note("Values are rounded to 3 decimal places at each step, as you would on paper.")

    # @state Weight updates
    text("## 6. Updating the weights")
    text("For every connection $i \\to j$: $\\;\\Delta w_{ij} = l \\cdot Err_j \\cdot O_i$.")
    O = {1: x[1], 2: x[2], 3: x[3], 4: O4, 5: O5}
    Err = {4: Err4, 5: Err5, 6: Err6}
    new_w = {}  # @inspect new_w
    rows = []
    for name in ["w46", "w56", "w14", "w15", "w24", "w25", "w34", "w35"]:
        i, j = int(name[1]), int(name[2])
        new_w[name], dw = update_weight(w[name], l, Err[j], O[i])  # @inspect new_w
        rows.append([name, w[name], f"Err{j} = {Err[j]:.3f}", f"O{i} = {O[i]:.3f}", f"({l})({Err[j]:.3f})({O[i]:.3f}) = {dw:+.3f}", new_w[name]])
    table(rows, headers=["Weight", "Old", "Err_j", "O_i", "Δw = l·Err_j·O_i", "New"], caption="Weight updates")
    note("w24 and w25 do not change because x2 = 0: an input of zero contributed nothing, so it gets no blame.")

    # @state Bias updates
    text("## 7. Updating the biases")
    text("For every unit $j$: $\\;\\Delta b_j = l \\cdot Err_j$.")
    new_b = {}  # @inspect new_b
    rows = []
    for name in ["b6", "b5", "b4"]:
        j = int(name[1])
        new_b[name], db = update_bias(b[name], l, Err[j])  # @inspect new_b
        rows.append([name, b[name], f"Err{j} = {Err[j]:.3f}", f"({l})({Err[j]:.3f}) = {db:+.3f}", new_b[name]])
    table(rows, headers=["Bias", "Old", "Err_j", "Δb = l·Err_j", "New"], caption="Bias updates")

    # @state Check
    text("## 8. Did it help?")
    text("Run the forward pass again with the updated parameters:")
    O4n = sigmoid(net_input([x[1], x[2], x[3]], [new_w["w14"], new_w["w24"], new_w["w34"]], new_b["b4"]))  # @stepover
    O5n = sigmoid(net_input([x[1], x[2], x[3]], [new_w["w15"], new_w["w25"], new_w["w35"]], new_b["b5"]))  # @stepover
    O6n = sigmoid(net_input([O4n, O5n], [new_w["w46"], new_w["w56"]], new_b["b6"]))  # @stepover @inspect O6n
    table(
        [["Before update", round(O6, 3), round(T - O6, 3)], ["After one update", round(O6n, 3), round(T - O6n, 3)]],
        headers=["", "Output O6", "Error T − O6"],
    )
    text(
        f"The output rose from **{O6:.3f}** to **{O6n:.3f}**, closer to the target of 1. "
        "One update moved the network in the right direction."
    )

    # @state Wrap-up
    text("## 9. Training in practice")
    text("**When to update:**")
    text(
        "- **Case updating**: update after each training example (what we did here).\n"
        "- **Epoch updating**: add up the changes and update once after the whole training set has been presented."
    )
    text("**When to stop:**")
    text(
        "- the change in the weights (or the error) falls below a threshold,\n"
        "- a set number of epochs has run, or\n"
        "- the share of examples misclassified in the previous epoch is low enough."
    )
    note("The learning rate sets the step size. Too large and training overshoots; too small and it crawls, or settles in a poor local minimum.")


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
