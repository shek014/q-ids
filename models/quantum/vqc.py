import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


def _one_hot(y, n_classes):
    Y = np.zeros((len(y), n_classes))
    Y[np.arange(len(y)), y] = 1.0
    return Y


class VQC:
    """Variational quantum classifier. Angle-embeds features onto n_qubits (one feature
    per qubit — see README > The dimensionality constraint), applies StronglyEntanglingLayers,
    and reads out one Z-expectation per class from the first n_classes wires -> softmax ->
    categorical cross-entropy. Mirrors the classical MLP's output stage so the two are
    compared on the same multiclass task rather than a scaled-down binary one.
    """

    def __init__(self, n_qubits, n_classes, n_layers=4, learning_rate=0.05, seed=None):
        if n_classes > n_qubits:
            raise ValueError("n_qubits must be >= n_classes (one readout qubit per class)")
        self.n_qubits = n_qubits
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)

        rng = np.random.default_rng(seed)
        init_weights = rng.normal(0.0, 0.1, size=(n_layers, n_qubits, 3))
        self.weights = pnp.array(init_weights, requires_grad=True)
        self.opt = qml.AdamOptimizer(stepsize=learning_rate)

        @qml.qnode(self.dev)
        def circuit(x, weights):
            qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_classes)]

        self._circuit = circuit

    def _probs(self, X, weights):
        raw = self._circuit(X, weights)  # list of n_classes arrays, each (batch,)
        logits = pnp.stack(raw, axis=1) * 2.0  # widen the [-1, 1] expval range before softmax
        shifted = logits - pnp.max(logits, axis=1, keepdims=True)
        exp = pnp.exp(shifted)
        return exp / pnp.sum(exp, axis=1, keepdims=True)

    def predict_proba(self, X):
        return np.asarray(self._probs(pnp.array(X), self.weights))

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def param_count(self):
        return self.weights.size

    def _cost(self, weights, X, Y_onehot):
        probs = self._probs(X, weights)
        eps = 1e-9
        return -pnp.mean(pnp.sum(Y_onehot * pnp.log(probs + eps), axis=1))

    def fit(self, X, y, X_val=None, y_val=None, epochs=30, batch_size=16,
            patience=None, verbose=True, seed=None):
        rng = np.random.default_rng(seed)
        n = X.shape[0]
        Y = _one_hot(y, self.n_classes)
        Xp, Yp = pnp.array(X), pnp.array(Y)

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        best_val_loss = np.inf
        best_weights = None
        epochs_without_improvement = 0

        for epoch in range(epochs):
            perm = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                xb, yb = Xp[idx], Yp[idx]
                self.weights = self.opt.step(lambda w: self._cost(w, xb, yb), self.weights)

            train_loss, train_acc = self._evaluate(X, y, Y)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            if X_val is not None:
                Y_val = _one_hot(y_val, self.n_classes)
                val_loss, val_acc = self._evaluate(X_val, y_val, Y_val)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = pnp.array(self.weights)
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if patience is not None and epochs_without_improvement >= patience:
                    if verbose:
                        print(f"early stopping at epoch {epoch + 1} (best val_loss={best_val_loss:.4f})")
                    break

            if verbose:
                msg = f"epoch {epoch + 1}/{epochs}  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}"
                if X_val is not None:
                    msg += f"  val_loss={history['val_loss'][-1]:.4f}  val_acc={history['val_acc'][-1]:.4f}"
                print(msg)

        if best_weights is not None:
            self.weights = best_weights

        return history

    def _evaluate(self, X, y, Y_onehot):
        probs = self.predict_proba(X)
        eps = 1e-9
        loss = -np.mean(np.sum(Y_onehot * np.log(probs + eps), axis=1))
        acc = np.mean(np.argmax(probs, axis=1) == y)
        return float(loss), float(acc)

    def save(self, path):
        np.savez(path, weights=np.asarray(self.weights),
                  n_qubits=self.n_qubits, n_classes=self.n_classes, n_layers=self.n_layers)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        model = cls(int(data["n_qubits"]), int(data["n_classes"]), int(data["n_layers"]))
        model.weights = pnp.array(data["weights"], requires_grad=True)
        return model
