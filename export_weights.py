"""Run this ONCE on your Mac (where TensorFlow installs fine).

It converts the Keras model into plain NumPy weights, so the Streamlit app
does not need TensorFlow at all. It creates two files next to this script:
    asl_weights.npz
    labels.json
Commit both to your GitHub repo.

Needs (locally only):  pip install tensorflow keras huggingface_hub
"""
import json
import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
import numpy as np
from huggingface_hub import hf_hub_download

from asl_numpy import NumpyMLP

REPO_ID = "AmimulBmeIU/asl-alphabet-classifier"


def fetch(filename):
    if os.path.exists(filename):
        return filename
    try:
        return hf_hub_download(REPO_ID, filename)
    except Exception:
        return hf_hub_download(REPO_ID, f"tfjs/{filename}")


def act_name(layer):
    act = getattr(layer, "activation", None)
    return getattr(act, "__name__", "linear") if act is not None else "linear"


model = keras.saving.load_model(fetch("model.keras"))
model.summary()

ops, arrays = [], {}
for layer in model.layers:
    kind = layer.__class__.__name__
    i = len(ops)
    if kind in ("InputLayer", "Dropout", "GaussianNoise", "AlphaDropout"):
        continue
    if kind == "Dense":
        has_bias = getattr(layer, "bias", None) is not None
        ops.append({"type": "dense", "activation": act_name(layer), "bias": has_bias})
        arrays[f"{i}_kernel"] = np.asarray(layer.kernel, dtype=np.float32)
        if has_bias:
            arrays[f"{i}_bias"] = np.asarray(layer.bias, dtype=np.float32)
    elif kind == "BatchNormalization":
        mean = np.asarray(layer.moving_mean, dtype=np.float32)
        var = np.asarray(layer.moving_variance, dtype=np.float32)
        gamma = layer.gamma
        beta = layer.beta
        ops.append({"type": "batchnorm", "eps": float(layer.epsilon)})
        arrays[f"{i}_mean"] = mean
        arrays[f"{i}_var"] = var
        arrays[f"{i}_gamma"] = (
            np.asarray(gamma, dtype=np.float32) if gamma is not None else np.ones_like(mean)
        )
        arrays[f"{i}_beta"] = (
            np.asarray(beta, dtype=np.float32) if beta is not None else np.zeros_like(mean)
        )
    elif kind == "Normalization":
        ops.append({"type": "normalization"})
        arrays[f"{i}_mean"] = np.asarray(layer.mean, dtype=np.float32)
        arrays[f"{i}_var"] = np.asarray(layer.variance, dtype=np.float32)
    elif kind in ("Activation", "ReLU", "Softmax"):
        name = {"ReLU": "relu", "Softmax": "softmax"}.get(kind) or act_name(layer)
        ops.append({"type": "activation", "activation": name})
    else:
        raise SystemExit(
            f"Layer type '{kind}' is not supported by the NumPy runner. "
            "Send me the model.summary() output above and I will add it."
        )

np.savez("asl_weights.npz", ops=np.array(json.dumps(ops)), **arrays)

with open(fetch("labels.json")) as f:
    labels = json.load(f)
with open("labels.json", "w") as f:
    json.dump(labels, f)

# Check: NumPy output must match Keras output
x = np.random.rand(50, 63).astype("float32")
expected = np.asarray(model(x, training=False))
got = NumpyMLP("asl_weights.npz").predict(x)
diff = float(np.abs(expected - got).max())
print(f"\nMax difference between Keras and NumPy outputs: {diff:.2e}")
if diff < 1e-4:
    print("OK - saved asl_weights.npz and labels.json. Commit both to GitHub.")
else:
    print("WARNING: outputs differ. Do NOT use these files; send me the summary above.")
