import numpy as np


def relu(z):
    return np.maximum(z, 0.0)


def relu_grad(z):
    return (z > 0).astype(z.dtype)


def softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def one_hot(y, n_classes):
    Y = np.zeros((y.shape[0], n_classes))
    Y[np.arange(y.shape[0]), y] = 1.0
    return Y


class Dense:
    def __init__(self, n_in, n_out, activation=None, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        # He init: right scale for ReLU-fed layers, a reasonable default for the linear output layer too.
        self.W = rng.normal(0.0, np.sqrt(2.0 / n_in), size=(n_in, n_out))
        self.b = np.zeros(n_out)
        self.activation = activation
        self._x = None
        self._z = None

    def forward(self, x):
        self._x = x
        self._z = x @ self.W + self.b
        return relu(self._z) if self.activation == "relu" else self._z

    def backward(self, d_out):
        dz = d_out * relu_grad(self._z) if self.activation == "relu" else d_out
        m = self._x.shape[0]
        dW = self._x.T @ dz / m
        db = np.mean(dz, axis=0)
        dx = dz @ self.W.T
        return dx, dW, db


class SGD:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self._velocity = None

    def step(self, params, grads):
        if self._velocity is None:
            self._velocity = [np.zeros_like(p) for p in params]
        for p, g, v in zip(params, grads, self._velocity):
            v *= self.momentum
            v -= self.lr * g
            p += v


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self._m = None
        self._v = None
        self._t = 0

    def step(self, params, grads):
        if self._m is None:
            self._m = [np.zeros_like(p) for p in params]
            self._v = [np.zeros_like(p) for p in params]
        self._t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self._m[i] = self.beta1 * self._m[i] + (1 - self.beta1) * g
            self._v[i] = self.beta2 * self._v[i] + (1 - self.beta2) * (g ** 2)
            m_hat = self._m[i] / (1 - self.beta1 ** self._t)
            v_hat = self._v[i] / (1 - self.beta2 ** self._t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class MLP:
    """layer_sizes e.g. [n_features, 16, 8, n_classes]. Output layer is linear;
    softmax + cross-entropy are combined in fit()/_evaluate() for a stable gradient."""

    def __init__(self, layer_sizes, optimizer="adam", learning_rate=1e-3, seed=None):
        rng = np.random.default_rng(seed)
        n_layers = len(layer_sizes) - 1
        self.layer_sizes = list(layer_sizes)
        self.layers = [
            Dense(layer_sizes[i], layer_sizes[i + 1],
                  activation="relu" if i < n_layers - 1 else None,
                  rng=rng)
            for i in range(n_layers)
        ]
        self.optimizer = Adam(lr=learning_rate) if optimizer == "adam" else SGD(lr=learning_rate)
        self.n_classes = layer_sizes[-1]

    def _forward(self, X):
        a = X
        for layer in self.layers:
            a = layer.forward(a)
        return a  # logits

    def _params(self):
        params = []
        for layer in self.layers:
            params.append(layer.W)
            params.append(layer.b)
        return params

    def predict_proba(self, X):
        return softmax(self._forward(X))

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def param_count(self):
        return sum(p.size for p in self._params())

    def fit(self, X, y, X_val=None, y_val=None, epochs=100, batch_size=32,
            class_weight=None, patience=None, verbose=True, seed=None):
        rng = np.random.default_rng(seed)
        n = X.shape[0]
        Y = one_hot(y, self.n_classes)
        sample_w = None
        if class_weight is not None:
            sample_w = np.array([class_weight[label] for label in y])

        Y_val = one_hot(y_val, self.n_classes) if X_val is not None else None

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        best_val_loss = np.inf
        best_state = None
        epochs_without_improvement = 0

        for epoch in range(epochs):
            perm = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                xb, yb = X[idx], Y[idx]
                wb = sample_w[idx] if sample_w is not None else None

                logits = self._forward(xb)
                probs = softmax(logits)

                d_logits = probs - yb
                if wb is not None:
                    d_logits *= wb[:, None]
                d_logits /= xb.shape[0]

                grads = []
                d = d_logits
                for layer in reversed(self.layers):
                    d, dW, db = layer.backward(d)
                    grads.append((dW, db))
                grads.reverse()
                flat_grads = [g for pair in grads for g in pair]

                self.optimizer.step(self._params(), flat_grads)

            train_loss, train_acc = self._evaluate(X, y, Y, sample_w)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            if X_val is not None:
                val_loss, val_acc = self._evaluate(X_val, y_val, Y_val)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = [p.copy() for p in self._params()]
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if patience is not None and epochs_without_improvement >= patience:
                    if verbose:
                        print(f"early stopping at epoch {epoch + 1} (best val_loss={best_val_loss:.4f})")
                    break

            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                msg = f"epoch {epoch + 1}/{epochs}  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}"
                if X_val is not None:
                    msg += f"  val_loss={history['val_loss'][-1]:.4f}  val_acc={history['val_acc'][-1]:.4f}"
                print(msg)

        if best_state is not None:
            for p, best in zip(self._params(), best_state):
                p[...] = best

        return history

    def _evaluate(self, X, y, Y_onehot, sample_w=None):
        probs = self.predict_proba(X)
        eps = 1e-12
        losses = -np.sum(Y_onehot * np.log(probs + eps), axis=1)
        loss = np.average(losses, weights=sample_w) if sample_w is not None else np.mean(losses)
        acc = np.mean(np.argmax(probs, axis=1) == y)
        return loss, acc

    def save(self, path):
        arrays = {}
        for i, layer in enumerate(self.layers):
            arrays[f"W{i}"] = layer.W
            arrays[f"b{i}"] = layer.b
        arrays["layer_sizes"] = np.array(self.layer_sizes)
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        layer_sizes = data["layer_sizes"].tolist()
        model = cls(layer_sizes)
        for i, layer in enumerate(model.layers):
            layer.W = data[f"W{i}"]
            layer.b = data[f"b{i}"]
        return model


if __name__ == "__main__":
    # sanity check on synthetic data — no sklearn, per project constraints
    rng = np.random.default_rng(0)
    n_per_class = 200
    X0 = rng.normal(loc=[-2, -2], scale=0.8, size=(n_per_class, 2))
    X1 = rng.normal(loc=[2, 2], scale=0.8, size=(n_per_class, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * n_per_class + [1] * n_per_class)

    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = MLP(layer_sizes=[2, 8, 8, 2], optimizer="adam", learning_rate=0.01, seed=0)
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val, epochs=50, batch_size=32, patience=10)
    print("params:", model.param_count())
