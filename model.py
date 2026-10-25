"""
Neural Networks From Scratch: Forward and Backward

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - numerical_gradient
import numpy as np

def numerical_gradient(f, x, eps=1e-5):
    grad = np.zeros_like(x, dtype=np.float64)
    if x.size == 0:
        return grad
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        orig_val = x[idx]
        x[idx] = orig_val + eps
        f_plus = f(x)
        x[idx] = orig_val - eps
        f_minus = f(x)
        x[idx] = orig_val
        grad[idx] = (f_plus - f_minus) / (2.0 * eps)
        it.iternext()
    return grad

# Step 2 - gradient_check
import numpy as np

def gradient_check(analytic_grad, numeric_grad, tol=1e-5):
    a = np.asarray(analytic_grad, dtype=np.float64)
    n = np.asarray(numeric_grad, dtype=np.float64)
    if a.size == 0:
        return 0.0
    denom = np.maximum(np.maximum(np.abs(a), np.abs(n)), tol)
    rel_err = np.abs(a - n) / denom
    return float(np.max(rel_err))

# Step 3 - make_dense
import numpy as np

def make_dense(in_dim, out_dim, weight_init_fn):
    W, b = weight_init_fn(in_dim, out_dim)
    params = {
        'W': np.array(W, dtype=np.float64, copy=True),
        'b': np.array(b, dtype=np.float64, copy=True)
    }

    def forward(x):
        cache = x
        y = x @ params['W'] + params['b']
        return y, cache

    def backward(dout, cache):
        x = cache
        dx = dout @ params['W'].T
        dW = x.T @ dout
        db = np.sum(dout, axis=0)
        return dx, {'W': dW, 'b': db}

    return {'forward': forward, 'backward': backward, 'params': params}

# Step 4 - make_activation
import numpy as np

def make_activation(kind='relu'):
    params = {}

    if kind == 'relu':
        def forward(x):
            cache = x
            y = np.maximum(0.0, x)
            return y, cache

        def backward(dout, cache):
            x = cache
            dx = dout * (x > 0).astype(dout.dtype)
            return dx, {}
    elif kind == 'tanh':
        def forward(x):
            y = np.tanh(x)
            cache = y
            return y, cache

        def backward(dout, cache):
            y = cache
            dx = dout * (1.0 - y ** 2)
            return dx, {}
    elif kind == 'sigmoid':
        def forward(x):
            y = 1.0 / (1.0 + np.exp(-x))
            cache = y
            return y, cache

        def backward(dout, cache):
            y = cache
            dx = dout * y * (1.0 - y)
            return dx, {}
    else:
        raise ValueError(f"Unsupported activation kind: {kind}")

    return {'forward': forward, 'backward': backward, 'params': params}

# Step 5 - initialize_weights
import numpy as np

def initialize_weights(in_dim, out_dim, scheme='he'):
    if scheme == 'he':
        std = np.sqrt(2.0 / in_dim)
        W = np.random.randn(in_dim, out_dim) * std
    elif scheme == 'xavier':
        std = np.sqrt(2.0 / (in_dim + out_dim))
        W = np.random.randn(in_dim, out_dim) * std
    else:
        raise ValueError(f"Unknown scheme: {scheme}")
    b = np.zeros(out_dim, dtype=np.float64)
    return W.astype(np.float64), b

# Step 6 - make_loss (not yet solved)
# TODO: implement

# Step 7 - make_sequential (not yet solved)
# TODO: implement

# Step 8 - forward_backward (not yet solved)
# TODO: implement

# Step 9 - make_optimizer (not yet solved)
# TODO: implement

# Step 10 - train_step (not yet solved)
# TODO: implement

# Step 11 - train (not yet solved)
# TODO: implement

# Step 12 - design_network (not yet solved)
# TODO: implement

# Step 13 - improve_generalization (not yet solved)
# TODO: implement

