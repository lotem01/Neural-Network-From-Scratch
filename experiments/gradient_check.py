"""Create the gradient test plot required by the assignment (notes, section 2.10.1)."""

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
    # Matrix inner product <d, G> = trace(d^T G) (notes, section 2.10.5).
    directional_derivative = np.sum(gradient * direction)

    epsilon = 0.1
    iterations = np.arange(1, 9)
    zero_order_errors = []
    first_order_errors = []
    for k in iterations:
        epsilon_k = epsilon * 0.5**k
        perturbed_loss, _ = loss_and_gradient(weights + epsilon_k * direction, features, targets)
        zero_order_errors.append(abs(perturbed_loss - loss))
        first_order_errors.append(abs(perturbed_loss - loss - epsilon_k * directional_derivative))

    print("k\terror order 1\t\terror order 2")
    for k, zero, first in zip(iterations, zero_order_errors, first_order_errors):
        print(f"{k}\t{zero:.6e}\t\t{first:.6e}")

    plt.semilogy(iterations, zero_order_errors, "o-", label="Zero order approx")
    plt.semilogy(iterations, first_order_errors, "o-", label="First order approx")
    plt.xlabel("k")
    plt.ylabel("error")
    plt.title("Successful Grad test in semilogarithmic plot")
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
