"""Tiny NumPy-only runner for the exported ASL landmark model (no TensorFlow needed)."""
import json

import numpy as np


def _act(x, name):
    if name in (None, "linear"):
        return x
    if name == "relu":
        return np.maximum(x, 0)
    if name == "softmax":
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)
    if name == "tanh":
        return np.tanh(x)
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    if name in ("swish", "silu"):
        return x / (1.0 + np.exp(-x))
    if name == "elu":
        return np.where(x > 0, x, np.exp(np.minimum(x, 0)) - 1.0)
    raise ValueError(f"Unsupported activation: {name}")


class NumpyMLP:
    def __init__(self, path):
        data = np.load(path, allow_pickle=False)
        self.ops = json.loads(str(data["ops"]))
        self.w = {k: data[k] for k in data.files if k != "ops"}

    def predict(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        for i, op in enumerate(self.ops):
            kind = op["type"]
            if kind == "dense":
                x = x @ self.w[f"{i}_kernel"]
                if op.get("bias"):
                    x = x + self.w[f"{i}_bias"]
                x = _act(x, op.get("activation"))
            elif kind == "batchnorm":
                x = (x - self.w[f"{i}_mean"]) / np.sqrt(self.w[f"{i}_var"] + op["eps"])
                x = x * self.w[f"{i}_gamma"] + self.w[f"{i}_beta"]
            elif kind == "normalization":
                x = (x - self.w[f"{i}_mean"]) / np.maximum(np.sqrt(self.w[f"{i}_var"]), 1e-7)
            elif kind == "activation":
                x = _act(x, op["activation"])
            else:
                raise ValueError(f"Unsupported op: {kind}")
        if self.ops[-1].get("activation") != "softmax":
            x = _act(x, "softmax")
        return x
