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

# Step 3 - make_dense (not yet solved)
# TODO: implement

# Step 4 - make_activation (not yet solved)
# TODO: implement

# Step 5 - initialize_weights (not yet solved)
# TODO: implement

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

