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

# Step 6 - make_loss
import numpy as np

def make_loss(kind='cross_entropy'):
    if kind != 'cross_entropy':
        raise ValueError(f"Unsupported loss: {kind}")

    def loss_fn(logits, labels):
        logits = np.asarray(logits, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        batch_size = logits.shape[0]

        # Log-sum-exp 保证数值稳定性
        max_logits = np.max(logits, axis=1, keepdims=True)
        shifted_logits = logits - max_logits
        exp_logits = np.exp(shifted_logits)
        sum_exp = np.sum(exp_logits, axis=1, keepdims=True)
        log_probs = shifted_logits - np.log(sum_exp)

        # 批次平均交叉熵
        correct_log_probs = log_probs[np.arange(batch_size), labels]
        loss = float(-np.mean(correct_log_probs))

        # Softmax 概率分布与梯度
        probs = exp_logits / sum_exp
        d_logits = probs.copy()
        d_logits[np.arange(batch_size), labels] -= 1.0
        d_logits /= batch_size

        return loss, d_logits

    return loss_fn

# Step 7 - make_sequential
import numpy as np

def make_dense(in_dim, out_dim, weight_init_fn):
    W, b = weight_init_fn(in_dim, out_dim)
    params = {'W': np.array(W, dtype=np.float64, copy=True),
              'b': np.array(b, dtype=np.float64, copy=True)}

    def forward(x):
        return x @ params['W'] + params['b'], x

    def backward(dout, cache):
        dx = dout @ params['W'].T
        dW = cache.T @ dout
        db = np.sum(dout, axis=0)
        return dx, {'W': dW, 'b': db}

    return {'forward': forward, 'backward': backward, 'params': params}

def make_activation(kind='relu'):
    params = {}
    if kind == 'relu':
        def forward(x):
            return np.maximum(0.0, x), x
        def backward(dout, cache):
            return dout * (cache > 0).astype(dout.dtype), {}
    elif kind == 'tanh':
        def forward(x):
            y = np.tanh(x)
            return y, y
        def backward(dout, cache):
            return dout * (1.0 - cache ** 2), {}
    else:
        raise ValueError(f"Unsupported kind: {kind}")
    return {'forward': forward, 'backward': backward, 'params': params}

def make_sequential(layers):
    params = [layer['params'] for layer in layers]

    def forward(x):
        caches, out = [], x
        for layer in layers:
            out, c = layer['forward'](out)
            caches.append(c)
        return out, caches

    def backward(dout, caches):
        grads_list = []
        cur_dout = dout
        for layer, c in zip(reversed(layers), reversed(caches)):
            b_func = layer['backward']
            cur_dout, g = b_func(cur_dout, c)
            grads_list.append(g)
        grads_list.reverse()
        return cur_dout, grads_list

    return {'forward': forward, 'backward': backward, 'params': params}

# Step 8 - forward_backward
def forward_backward(model, loss_fn, x, y):
    logits, caches = model['forward'](x)
    loss, d_logits = loss_fn(logits, y)
    _, param_grads = model['backward'](<d_logits, caches>)
    return float(loss), param_grads

# Step 9 - make_optimizer
import numpy as np

def make_optimizer(params, lr=1e-2, kind='sgd'):
    if kind != 'sgd':
        raise ValueError(f"Unsupported optimizer kind: {kind}")

    def update_rec(p_node, g_node):
        if isinstance(p_node, dict):
            for k in p_node:
                if k in g_node:
                    update_rec(p_node[k], g_node[k])
        elif isinstance(p_node, list):
            for p_sub, g_sub in zip(p_node, g_node):
                update_rec(p_sub, g_sub)
        elif isinstance(p_node, np.ndarray):
            p_node -= lr * g_node

    def step(grads):
        update_rec(params, grads)

    return {'step': step, 'lr': lr, 'kind': kind}

# Step 10 - train_step
def train_step(model, loss_fn, optimizer, x_batch, y_batch):
    loss, grads = forward_backward(model, loss_fn, x_batch, y_batch)
    optimizer['step'](grads)
    return float(loss)

# Step 11 - train
import numpy as np

def train(model, loss_fn, optimizer, x, y, epochs, batch_size, seed=0):
    rng = np.random.RandomState(seed)
    n = len(x)
    history = []
    for epoch in range(epochs):
        perm = rng.permutation(n)
        epoch_losses = []
        for i in range(0, n, batch_size):
            batch_idx = perm[i:i + batch_size]
            xb = x[batch_idx]
            yb = y[batch_idx]
            l = train_step(model, loss_fn, optimizer, xb, yb)
            epoch_losses.append(l)
        history.append(float(np.mean(epoch_losses)))
    return history

# Step 12 - design_network
import numpy as np

def design_network(input_dim, num_classes, seed=0):
    pts_per_class = 60
    rng = np.random.RandomState(seed)
    xs, ys = [], []
    for c in range(num_classes):
        r_base = 1.0 + 1.8 * c
        v = rng.randn(pts_per_class, input_dim)
        norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-10
        u = v / norms
        r = r_base + rng.uniform(-0.2, 0.2, size=(pts_per_class, 1))
        xc = u * r
        xs.append(xc)
        ys.append(np.full(pts_per_class, c, dtype=np.int64))
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    perm = rng.permutation(len(y))
    x, y = x[perm], y[perm]

    def init_fn(i, o):
        return initialize_weights(i, o, scheme='he')

    layers = [
        make_dense(input_dim, 64, init_fn),
        make_activation('relu'),
        make_dense(64, 64, init_fn),
        make_activation('relu'),
        make_dense(64, num_classes, init_fn),
    ]
    model = make_sequential(layers)
    loss_fn = make_loss('cross_entropy')
    opt = make_optimizer(model['params'], lr=0.1, kind='sgd')
    train(model, loss_fn, opt, x, y, epochs=150, batch_size=16, seed=seed)

    logits, _ = model['forward'](x)
    preds = np.argmax(np.asarray(logits), axis=1)
    acc = float(np.mean(preds == y))

    metrics = {
        'accuracy': acc,
        'x': x,
        'y': y
    }
    return model, metrics

# Step 13 - improve_generalization
import numpy as np

def improve_generalization(baseline_model_fn, x_train, y_train, x_val, y_val, seed=0):
    # 1. 评估无正则化基线模型
    base_model = baseline_model_fn()
    base_loss_fn = make_loss('cross_entropy')
    base_params = base_model['params']
    def base_step(grads):
        for p, g in zip(base_params, grads):
            if isinstance(p, dict) and isinstance(g, dict):
                for k in p:
                    if k in g and isinstance(p[k], np.ndarray):
                        p[k] -= 0.05 * np.asarray(g[k], dtype=p[k].dtype)
            elif isinstance(p, np.ndarray):
                p -= 0.05 * np.asarray(g, dtype=p.dtype)
    base_opt = {'step': base_step, 'lr': 0.05, 'kind': 'sgd'}
    train(base_model, base_loss_fn, base_opt, x_train, y_train, epochs=70, batch_size=16, seed=seed)
    base_logits, _ = base_model['forward'](x_val)
    base_preds = np.argmax(np.asarray(base_logits), axis=1)
    baseline_val_accuracy = float(np.mean(base_preds == y_val))

    # 2. 训练改进模型（引入权重衰减 + 早停控制）
    model = baseline_model_fn()
    loss_fn = make_loss('cross_entropy')
    lr = 0.05
    decay = 0.03
    params = model['params']
    def step(grads):
        for p, g in zip(params, grads):
            if isinstance(p, dict) and isinstance(g, dict):
                for k in p:
                    if k in g and isinstance(p[k], np.ndarray):
                        if k == 'W':
                            p[k] *= (1.0 - lr * decay)
                        p[k] -= lr * np.asarray(g[k], dtype=p[k].dtype)
            elif isinstance(p, np.ndarray):
                p -= lr * np.asarray(g, dtype=p.dtype)

    opt = {'step': step, 'lr': lr, 'kind': 'sgd'}
    train(model, loss_fn, opt, x_train, y_train, epochs=45, batch_size=16, seed=seed)
    logits, _ = model['forward'](x_val)
    predictions = np.argmax(np.asarray(logits), axis=1).astype(int)
    val_accuracy = float(np.mean(predictions == y_val))

    return {
        'val_accuracy': val_accuracy,
        'baseline_val_accuracy': baseline_val_accuracy,
        'predictions': predictions,
        'model': model
    }

