"""Create the directional gradient test plot required by the assignment."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from softmax import add_bias, loss_and_gradient


def main():
    rng = np.random.default_rng(42)
    features = add_bias(rng.standard_normal((4, 30)))
    targets = np.eye(3)[:, rng.integers(0, 3, size=features.shape[1])]
    weights = rng.standard_normal((features.shape[0], targets.shape[0]))
    direction = rng.standard_normal(weights.shape)
    direction /= np.linalg.norm(direction)

    loss, gradient = loss_and_gradient(weights, features, targets)
    directional_derivative = np.sum(gradient * direction)
    steps = 0.5 ** np.arange(1, 13)
    first_order_errors = []
    second_order_errors = []
    for step in steps:
        perturbed_loss, _ = loss_and_gradient(weights + step * direction, features, targets)
        first_order_errors.append(abs(perturbed_loss - loss))
        second_order_errors.append(abs(perturbed_loss - loss - step * directional_derivative))

    print("step\t\t|F(W + hD) - F(W)|\tTaylor remainder")
    for step, first, second in zip(steps, first_order_errors, second_order_errors):
        print(f"{step:.6e}\t{first:.6e}\t\t{second:.6e}")

    plt.loglog(steps, first_order_errors, "o-", label="Loss change: O(h)")
    plt.loglog(steps, second_order_errors, "o-", label="Taylor remainder: O(h²)")
    plt.loglog(steps, first_order_errors[0] * steps / steps[0], "--", label="Reference slope 1")
    plt.loglog(steps, second_order_errors[0] * (steps / steps[0]) ** 2, "--", label="Reference slope 2")
    plt.xlabel("Step size h")
    plt.ylabel("Absolute error")
    plt.title("Softmax directional gradient test")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    output = Path("results/softmax_gradient_check.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=160)
    plt.close()
    print(f"Plot saved to {output}")


if __name__ == "__main__":
    main()
