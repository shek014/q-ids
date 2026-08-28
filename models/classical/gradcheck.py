"""Numerical gradient check for the hand-written MLP backprop.

Confirms the analytical gradients the training loop feeds to the optimizer match a
finite-difference estimate of the mean cross-entropy loss's gradient. This is the one
thing "the loss goes down" does NOT prove: a sign or transpose error can still converge
while being wrong. Run: python -m models.classical.gradcheck
"""
import numpy as np

from models.classical.mlp import MLP, one_hot, softmax


def mean_ce_loss(model, X, Y_onehot):
    probs = softmax(model._forward(X))
    return -np.mean(np.sum(Y_onehot * np.log(probs + 1e-12), axis=1))


def numerical_grads(model, X, Y_onehot, eps=1e-6):
    """Central-difference estimate of d(mean_ce_loss)/d(param) for every weight and bias."""
    grads = []
    for layer in model.layers:
        dW = np.zeros_like(layer.W)
        it = np.nditer(layer.W, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig = layer.W[idx]
            layer.W[idx] = orig + eps
            lp = mean_ce_loss(model, X, Y_onehot)
            layer.W[idx] = orig - eps
            lm = mean_ce_loss(model, X, Y_onehot)
            layer.W[idx] = orig
            dW[idx] = (lp - lm) / (2 * eps)
            it.iternext()

        db = np.zeros_like(layer.b)
        for i in range(len(layer.b)):
            orig = layer.b[i]
            layer.b[i] = orig + eps
            lp = mean_ce_loss(model, X, Y_onehot)
            layer.b[i] = orig - eps
            lm = mean_ce_loss(model, X, Y_onehot)
            layer.b[i] = orig
            db[i] = (lp - lm) / (2 * eps)

        grads.append((dW, db))
    return grads


def rel_error(a, b):
    return np.max(np.abs(a - b) / np.clip(np.abs(a) + np.abs(b), 1e-12, None))


def main():
    rng = np.random.default_rng(0)
    n, n_features, n_classes = 8, 5, 3
    X = rng.normal(size=(n, n_features))
    y = rng.integers(0, n_classes, size=n)
    Y = one_hot(y, n_classes)

    model = MLP([n_features, 6, 4, n_classes], seed=0)
    ana = model._grads(X, Y)            # the exact gradient the training loop uses
    num = numerical_grads(model, X, Y)

    worst = 0.0
    for i, ((aW, ab), (nW, nb)) in enumerate(zip(ana, num)):
        eW, eb = rel_error(aW, nW), rel_error(ab, nb)
        worst = max(worst, eW, eb)
        print(f"layer {i}: W rel_err={eW:.2e}  b rel_err={eb:.2e}")

    print(f"\nworst relative error: {worst:.2e}")
    ok = worst < 1e-5
    print("PASS" if ok else "FAIL — analytical gradients do not match numerical")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
