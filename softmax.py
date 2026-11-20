"""Softmax regression for data arranged as columns."""

import numpy as np


def add_bias(features):
    """Add a row of ones so the last row of the weights is the bias."""
    return np.vstack((features, np.ones((1, features.shape[1]))))


def loss_and_gradient(weights, features, targets):
    """Return mean cross-entropy and its gradient with respect to weights.

    features: (features including bias, samples)
    weights:  (features including bias, classes)
    targets:  (classes, samples), with each column summing to one
    """
    scores = weights.T @ features
    shifted_scores = scores - np.max(scores, axis=0, keepdims=True)
    log_probabilities = shifted_scores - np.log(
        np.sum(np.exp(shifted_scores), axis=0, keepdims=True)
    )
    probabilities = np.exp(log_probabilities)
    sample_count = features.shape[1]
    loss = -np.sum(targets * log_probabilities) / sample_count
    gradient = features @ (probabilities - targets).T / sample_count
    return loss, gradient
