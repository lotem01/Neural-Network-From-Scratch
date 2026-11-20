import unittest

import numpy as np

from softmax import add_bias, loss_and_gradient


class SoftmaxTests(unittest.TestCase):
    def test_uniform_scores_and_bias_gradient(self):
        features = add_bias(np.array([[2.0, -1.0]]))
        targets = np.array([[1.0, 1.0], [0.0, 0.0]])
        loss, gradient = loss_and_gradient(np.zeros((2, 2)), features, targets)
        self.assertAlmostEqual(loss, np.log(2))
        np.testing.assert_allclose(gradient, [[-0.25, 0.25], [-0.5, 0.5]])

    def test_large_scores_produce_finite_loss_and_gradient(self):
        features = add_bias(np.array([[1.0, -1.0]]))
        targets = np.array([[0.0, 1.0], [1.0, 0.0]])
        weights = np.array([[1000.0, -1000.0], [0.0, 0.0]])
        loss, gradient = loss_and_gradient(weights, features, targets)
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(np.isfinite(gradient).all())
        self.assertAlmostEqual(loss, 2000.0)

    def test_gradient_against_central_difference(self):
        rng = np.random.default_rng(7)
        features = add_bias(rng.normal(size=(3, 8)))
        targets = np.eye(3)[:, rng.integers(0, 3, size=8)]
        weights = rng.normal(size=(4, 3))
        _, gradient = loss_and_gradient(weights, features, targets)
        step = 1e-5
        for index in np.ndindex(weights.shape):
            direction = np.zeros_like(weights)
            direction[index] = step
            plus, _ = loss_and_gradient(weights + direction, features, targets)
            minus, _ = loss_and_gradient(weights - direction, features, targets)
            self.assertAlmostEqual(gradient[index], (plus - minus) / (2 * step), places=8)


if __name__ == "__main__":
    unittest.main()
