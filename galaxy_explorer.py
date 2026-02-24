#!/usr/bin/env python3
"""Activation Galaxy Explorer - Full pipeline + static PNG output.

Self-contained script that:
  1. Downloads any missing datasets (MNIST, FashionMNIST, EMNIST, CIFAR-10)
  2. Seeds evolution with common known activations (ReLU, sigmoid, tanh,
     swish, GELU, ELU, SELU, mish, leaky ReLU, softsign, softplus, etc.)
  3. Evolves new activations via genetic programming per dataset
  4. Collects all catalogs (existing + newly evolved)
  5. Deduplicates and cross-scores on all datasets
  6. Renders a t-SNE galaxy map as a PNG (no pygame / no display needed)

Usage:
    python galaxy_explorer.py                    # full pipeline
    python galaxy_explorer.py --catalog path.json  # skip evolution, just render
    python galaxy_explorer.py --quick            # smaller evolution (fast test)
"""

import os
import sys
import json
import time
import math
import copy
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
SEED = 42
TRAIN_FRAC = 0.8
N_CURVE_POINTS = 200  # Points in activation curve fingerprint

# Characterization
N_CHAR_NEURONS = 32
N_EVAL_BATCHES = 5
EVAL_BATCH_SIZE = 2000
KNN_K = 7

# Evolution (full)
POP_SIZE = 300
GENS = 200
MAX_DEPTH = 5
MAX_NODES = 25
TOURNAMENT_SIZE = 5
MUTATION_RATE = 0.7
SUBTREE_MUT_RATE = 0.4
POINT_MUT_RATE = 0.3
CONST_MUT_RATE = 0.2
HOIST_MUT_RATE = 0.1
N_RANDOM_EXTRA = 1000

# Quick mode overrides
QUICK_POP_SIZE = 60
QUICK_GENS = 30
QUICK_RANDOM_EXTRA = 100

# Primordial operations
UNARY_OPS = ['neg', 'abs', 'exp', 'log', 'sin', 'cos']
BINARY_OPS = ['+', '-', '*', '/']
CONSTANTS = [0.0, 0.5, 1.0, -1.0, 2.0, math.e, math.pi]

# Datasets to use for evolution + cross-scoring
DATASET_CONFIGS = [
    {
        'name': 'MNIST',
        'loader': 'MNIST',
        'input_dim': 784,
        'n_classes': 10,
        'flatten': True,
        'n_char_neurons': 32,
    },
    {
        'name': 'FashionMNIST',
        'loader': 'FashionMNIST',
        'input_dim': 784,
        'n_classes': 10,
        'flatten': True,
        'n_char_neurons': 32,
    },
    {
        'name': 'EMNIST_Digits',
        'loader': 'EMNIST',
        'loader_kwargs': {'split': 'digits'},
        'input_dim': 784,
        'n_classes': 10,
        'flatten': True,
        'max_samples': 60000,
        'n_char_neurons': 32,
    },
    {
        'name': 'CIFAR10',
        'loader': 'CIFAR10',
        'input_dim': 3072,
        'n_classes': 10,
        'flatten': True,
        'n_char_neurons': 96,
    },
]

# Task keys for cross-scoring (one per dataset)
TASK_KEYS = [cfg['name'] for cfg in DATASET_CONFIGS]

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════
_LOG_FILE = None


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode('ascii', errors='replace').decode('ascii'), flush=True)
    if _LOG_FILE:
        try:
            with open(_LOG_FILE, 'a', encoding='utf-8') as f:
                f.write(line + "\n")
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXPRESSION TREE
# ═══════════════════════════════════════════════════════════════════════════════
class Expr:
    def eval(self, x):
        raise NotImplementedError
    def depth(self):
        raise NotImplementedError
    def size(self):
        raise NotImplementedError
    def copy(self):
        return copy.deepcopy(self)
    def nodes(self):
        raise NotImplementedError


class Const(Expr):
    def __init__(self, value):
        self.value = float(value)

    def eval(self, x):
        return torch.full_like(x, self.value)

    def depth(self):
        return 0

    def size(self):
        return 1

    def nodes(self):
        return [self]

    def __str__(self):
        if self.value == math.e:
            return 'e'
        elif self.value == math.pi:
            return 'pi'
        elif self.value == int(self.value):
            return str(int(self.value))
        else:
            return f"{self.value:.3g}"


class Var(Expr):
    def eval(self, x):
        return x

    def depth(self):
        return 0

    def size(self):
        return 1

    def nodes(self):
        return [self]

    def __str__(self):
        return 'x'


class UnaryOp(Expr):
    def __init__(self, op, child):
        self.op = op
        self.child = child

    def eval(self, x):
        c = self.child.eval(x)
        c = torch.clamp(c, -50, 50)
        if self.op == 'neg':
            return -c
        elif self.op == 'abs':
            return torch.abs(c)
        elif self.op == 'exp':
            return torch.exp(torch.clamp(c, -20, 20))
        elif self.op == 'log':
            return torch.log(torch.abs(c) + 1e-8)
        elif self.op == 'sin':
            return torch.sin(c)
        elif self.op == 'cos':
            return torch.cos(c)
        else:
            return c

    def depth(self):
        return 1 + self.child.depth()

    def size(self):
        return 1 + self.child.size()

    def nodes(self):
        return [self] + self.child.nodes()

    def __str__(self):
        return f"{self.op}({self.child})"


class BinaryOp(Expr):
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right

    def eval(self, x):
        a = torch.clamp(self.left.eval(x), -50, 50)
        b = torch.clamp(self.right.eval(x), -50, 50)
        if self.op == '+':
            return a + b
        elif self.op == '-':
            return a - b
        elif self.op == '*':
            return a * b
        elif self.op == '/':
            sign_b = torch.sign(b)
            sign_b = torch.where(sign_b == 0, torch.ones_like(sign_b), sign_b)
            return a / (torch.abs(b).clamp(min=1e-6) * sign_b)
        else:
            return a

    def depth(self):
        return 1 + max(self.left.depth(), self.right.depth())

    def size(self):
        return 1 + self.left.size() + self.right.size()

    def nodes(self):
        return [self] + self.left.nodes() + self.right.nodes()

    def __str__(self):
        return f"({self.left} {self.op} {self.right})"


# ═══════════════════════════════════════════════════════════════════════════════
# RANDOM TREE GENERATION & GENETIC OPERATORS
# ═══════════════════════════════════════════════════════════════════════════════
def random_terminal():
    if random.random() < 0.55:
        return Var()
    else:
        return Const(random.choice(CONSTANTS))


def random_tree(max_depth, current_depth=0):
    if current_depth >= max_depth:
        return random_terminal()
    p_terminal = 0.2 + 0.15 * current_depth
    if current_depth > 0 and random.random() < p_terminal:
        return random_terminal()
    if random.random() < 0.45:
        op = random.choice(UNARY_OPS)
        child = random_tree(max_depth, current_depth + 1)
        return UnaryOp(op, child)
    else:
        op = random.choice(BINARY_OPS)
        left = random_tree(max_depth, current_depth + 1)
        right = random_tree(max_depth, current_depth + 1)
        return BinaryOp(op, left, right)


def replace_random_subtree(tree, new_subtree):
    tree = tree.copy()
    all_nodes = tree.nodes()
    if len(all_nodes) <= 1:
        return new_subtree.copy()
    target_idx = random.randint(1, len(all_nodes) - 1)
    target = all_nodes[target_idx]

    def _replace(node, target_id, replacement):
        if isinstance(node, UnaryOp):
            if id(node.child) == target_id:
                node.child = replacement
                return True
            return _replace(node.child, target_id, replacement)
        elif isinstance(node, BinaryOp):
            if id(node.left) == target_id:
                node.left = replacement
                return True
            if id(node.right) == target_id:
                node.right = replacement
                return True
            return _replace(node.left, target_id, replacement) or \
                   _replace(node.right, target_id, replacement)
        return False

    _replace(tree, id(target), new_subtree.copy())
    return tree


def mutate(tree):
    r = random.random()
    if r < SUBTREE_MUT_RATE:
        new_sub = random_tree(max_depth=3, current_depth=0)
        return replace_random_subtree(tree, new_sub)
    elif r < SUBTREE_MUT_RATE + POINT_MUT_RATE:
        tree = tree.copy()
        all_nodes = tree.nodes()
        candidates = [n for n in all_nodes if isinstance(n, (UnaryOp, BinaryOp))]
        if candidates:
            node = random.choice(candidates)
            if isinstance(node, UnaryOp):
                node.op = random.choice(UNARY_OPS)
            elif isinstance(node, BinaryOp):
                node.op = random.choice(BINARY_OPS)
        return tree
    elif r < SUBTREE_MUT_RATE + POINT_MUT_RATE + CONST_MUT_RATE:
        tree = tree.copy()
        all_nodes = tree.nodes()
        consts = [n for n in all_nodes if isinstance(n, Const)]
        if consts:
            node = random.choice(consts)
            node.value += random.gauss(0, 0.5)
        else:
            return replace_random_subtree(tree, Const(random.gauss(0, 1)))
        return tree
    else:
        all_nodes = tree.nodes()
        non_trivial = [n for n in all_nodes if isinstance(n, (UnaryOp, BinaryOp))]
        if non_trivial:
            return random.choice(non_trivial).copy()
        return tree.copy()


def crossover(tree1, tree2):
    child = tree1.copy()
    donor_nodes = tree2.nodes()
    if len(donor_nodes) > 0:
        donor_sub = random.choice(donor_nodes).copy()
        child = replace_random_subtree(child, donor_sub)
    return child


# ═══════════════════════════════════════════════════════════════════════════════
# SAFE EVAL & DEDUP
# ═══════════════════════════════════════════════════════════════════════════════
def safe_eval_curve(expr, z):
    try:
        result = expr.eval(z)
        result = torch.where(torch.isfinite(result), result, torch.zeros_like(result))
        result = torch.clamp(result, -50, 50)
        return result
    except Exception:
        return torch.zeros_like(z)


def curve_fingerprint(expr, n_points=200):
    z = torch.linspace(-5, 5, n_points)
    y = safe_eval_curve(expr, z)
    std = y.std().item()
    if std < 1e-6:
        return None, y.numpy()
    y_norm = (y - y.mean()) / (std + 1e-8)
    return y_norm.numpy(), y.numpy()


def is_duplicate(fingerprint, existing_fingerprints, threshold=0.999):
    if fingerprint is None:
        return False
    fp = fingerprint / (np.linalg.norm(fingerprint) + 1e-8)
    for efp in existing_fingerprints:
        if efp is None:
            continue
        enorm = efp / (np.linalg.norm(efp) + 1e-8)
        sim = np.dot(fp, enorm)
        if abs(sim) > threshold:
            return True
    return False


def deduplicate_by_curve(merged, threshold=0.999):
    """Remove near-duplicate curves using cosine similarity."""
    curves = torch.tensor([a['curve'] for a in merged], dtype=torch.float32)
    n = curves.shape[0]
    if n == 0:
        return merged

    norms = curves.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normed = curves / norms

    keep = [True] * n
    chunk = 2000
    for i in range(0, n, chunk):
        end_i = min(i + chunk, n)
        if not any(keep[i:end_i]):
            continue
        block = normed[i:end_i]
        for j in range(0, i, chunk):
            end_j = min(j + chunk, n)
            kept_j = [k for k in range(j, end_j) if keep[k]]
            if not kept_j:
                continue
            ref_kept = normed[kept_j]
            sim = block @ ref_kept.T
            dups = (sim > threshold).any(dim=1)
            for idx_in_block, is_dup in enumerate(dups):
                if is_dup:
                    keep[i + idx_in_block] = False

    return [a for a, k in zip(merged, keep) if k]


# ═══════════════════════════════════════════════════════════════════════════════
# KNOWN ACTIVATIONS (expanded set)
#
# These are common activation functions that would be very difficult to
# rediscover purely through evolution from primordial operations.
# ═══════════════════════════════════════════════════════════════════════════════
def build_known_activations():
    """Build a comprehensive set of known activation functions from expression trees.

    Includes all common activations that are hard to discover via evolution:
    - Basics: identity, abs, neg, square, sin, cos, exp
    - Standard ML: ReLU, sigmoid, tanh, softplus, softsign
    - Modern: Swish/SiLU, GELU (approx), Mish (approx), ELU (approx),
              SELU (approx), leaky ReLU, hard sigmoid, hard swish
    - Novel: gaussian, laplacian, x*sin(x), sin(x^2), cos(x^2), etc.
    """
    x = Var()
    c = lambda v: Const(v)

    # Helper: sigmoid(x) = 1 / (1 + exp(-x))
    def _sigmoid():
        return BinaryOp('/', c(1), BinaryOp('+', c(1), UnaryOp('exp', UnaryOp('neg', Var()))))

    # Helper: tanh(x) ≈ (exp(2x) - 1) / (exp(2x) + 1)
    def _tanh():
        exp2x = UnaryOp('exp', BinaryOp('*', c(2), Var()))
        return BinaryOp('/', BinaryOp('-', exp2x, c(1)), BinaryOp('+', exp2x, c(1)))

    activations = {
        # ── Basics ──────────────────────────────────────────────────────
        'identity': Var(),
        'abs': UnaryOp('abs', Var()),
        'neg': UnaryOp('neg', Var()),
        'square': BinaryOp('*', Var(), Var()),
        'cube': BinaryOp('*', Var(), BinaryOp('*', Var(), Var())),
        'exp': UnaryOp('exp', Var()),
        'sin': UnaryOp('sin', Var()),
        'cos': UnaryOp('cos', Var()),

        # ── Standard ML activations ─────────────────────────────────────
        # ReLU = max(0, x) = (x + |x|) * 0.5
        'relu': BinaryOp('*', BinaryOp('+', Var(), UnaryOp('abs', Var())), c(0.5)),

        # Leaky ReLU ≈ max(0.01x, x) = 0.5 * ((1 + 0.01) * x + (1 - 0.01) * |x|)
        #            simplified: 0.505 * x + 0.495 * |x|
        'leaky_relu': BinaryOp('+',
            BinaryOp('*', c(0.505), Var()),
            BinaryOp('*', c(0.495), UnaryOp('abs', Var()))),

        # Sigmoid = 1 / (1 + exp(-x))
        'sigmoid': _sigmoid(),

        # Tanh = (exp(2x) - 1) / (exp(2x) + 1)
        'tanh': _tanh(),

        # Softplus = log(1 + exp(x))
        'softplus': UnaryOp('log', BinaryOp('+', c(1), UnaryOp('exp', Var()))),

        # Softsign = x / (1 + |x|)
        'softsign': BinaryOp('/', Var(), BinaryOp('+', c(1), UnaryOp('abs', Var()))),

        # ── Modern activations ──────────────────────────────────────────
        # Swish / SiLU = x * sigmoid(x) = x / (1 + exp(-x))
        'swish': BinaryOp('*', Var(), _sigmoid()),

        # GELU ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        # Simplified approximation: x * sigmoid(1.702 * x)
        'gelu_approx': BinaryOp('*', Var(),
            BinaryOp('/', c(1),
                BinaryOp('+', c(1),
                    UnaryOp('exp', UnaryOp('neg',
                        BinaryOp('*', c(1.702), Var())))))),

        # Mish ≈ x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
        # We use the expression tree form
        'mish_approx': BinaryOp('*', Var(),
            _tanh_of(UnaryOp('log', BinaryOp('+', c(1), UnaryOp('exp', Var()))))),

        # ELU ≈ x if x > 0, exp(x) - 1 if x <= 0
        # Approximation: 0.5 * (x + |x|) + 0.5 * (exp(min(x, 0)) - 1)
        # Simpler: (x + |x|) * 0.5 + (exp(x - |x|) * 0.5 - 0.5) * 0.5... too complex
        # Use: relu(x) + (exp(-relu(-x)) - 1)
        # = 0.5*(x+|x|) + exp(-0.5*(|x|-x)) - 1
        'elu_approx': BinaryOp('+',
            BinaryOp('*', c(0.5), BinaryOp('+', Var(), UnaryOp('abs', Var()))),
            BinaryOp('-',
                UnaryOp('exp', BinaryOp('*', c(-0.5),
                    BinaryOp('-', UnaryOp('abs', Var()), Var()))),
                c(1))),

        # SELU ≈ 1.0507 * (max(0,x) + 1.6733 * (exp(min(0,x)) - 1))
        # Simplified: scale * elu(x)
        'selu_approx': BinaryOp('*', c(1.0507),
            BinaryOp('+',
                BinaryOp('*', c(0.5), BinaryOp('+', Var(), UnaryOp('abs', Var()))),
                BinaryOp('*', c(1.6733),
                    BinaryOp('-',
                        UnaryOp('exp', BinaryOp('*', c(-0.5),
                            BinaryOp('-', UnaryOp('abs', Var()), Var()))),
                        c(1))))),

        # Hard sigmoid ≈ clip(0.2*x + 0.5, 0, 1) ≈ relu6(x+3)/6
        # Approx: 0.5 + 0.5 * softsign(x)
        'hard_sigmoid': BinaryOp('+', c(0.5),
            BinaryOp('*', c(0.5),
                BinaryOp('/', Var(), BinaryOp('+', c(1), UnaryOp('abs', Var()))))),

        # Hard swish ≈ x * hard_sigmoid(x)
        'hard_swish': BinaryOp('*', Var(),
            BinaryOp('+', c(0.5),
                BinaryOp('*', c(0.5),
                    BinaryOp('/', Var(), BinaryOp('+', c(1), UnaryOp('abs', Var())))))),

        # ── Gaussian / bell-curve family ────────────────────────────────
        # Gaussian = exp(-x^2)
        'gaussian': UnaryOp('exp', UnaryOp('neg', BinaryOp('*', Var(), Var()))),

        # Laplacian = exp(-|x|)
        'laplacian': UnaryOp('exp', UnaryOp('neg', UnaryOp('abs', Var()))),

        # Bump = exp(-1/(1-x^2)) approx for |x|<1 ... simplified as:
        # Narrow gaussian: exp(-2*x^2)
        'narrow_gaussian': UnaryOp('exp',
            UnaryOp('neg', BinaryOp('*', c(2), BinaryOp('*', Var(), Var())))),

        # ── Compositional / novel ───────────────────────────────────────
        # x * sin(x) — oscillatory with linear envelope
        'x_sin_x': BinaryOp('*', Var(), UnaryOp('sin', Var())),

        # x * cos(x)
        'x_cos_x': BinaryOp('*', Var(), UnaryOp('cos', Var())),

        # sin(x^2) — chirp-like
        'sin_sq': UnaryOp('sin', BinaryOp('*', Var(), Var())),

        # cos(x^2)
        'cos_sq': UnaryOp('cos', BinaryOp('*', Var(), Var())),

        # Negative ReLU: max(0, -x)
        'neg_relu': BinaryOp('*', c(0.5),
            BinaryOp('-', UnaryOp('abs', Var()), Var())),

        # Sign function: x / (|x| + epsilon) ≈ softsign with steep slope
        'sharp_softsign': BinaryOp('/', Var(),
            BinaryOp('+', c(0.01), UnaryOp('abs', Var()))),

        # Sigmoid-weighted linear unit variant: x * (1 - sigmoid(-x))
        # = x * sigmoid(x) = swish  (skip, already have it)

        # Penalized tanh: tanh(x) for x>0, 0.25*tanh(x) for x<0
        # Approx: 0.625 * tanh(x) + 0.375 * tanh(x) * sign(x)
        # Simpler: just add tanh * relu combo
        'tanh_relu': BinaryOp('*',
            _tanh(),
            BinaryOp('*', c(0.5), BinaryOp('+', Var(), UnaryOp('abs', Var())))),

        # Absolute exponential decay: x * exp(-|x|)
        'x_exp_neg_abs': BinaryOp('*', Var(),
            UnaryOp('exp', UnaryOp('neg', UnaryOp('abs', Var())))),

        # Sine gate: sin(x) * sigmoid(x)
        'sin_gate': BinaryOp('*', UnaryOp('sin', Var()), _sigmoid()),

        # Log-cosh: log(cosh(x)) ≈ log((exp(x) + exp(-x)) / 2)
        'log_cosh': UnaryOp('log',
            BinaryOp('*', c(0.5),
                BinaryOp('+',
                    UnaryOp('exp', Var()),
                    UnaryOp('exp', UnaryOp('neg', Var()))))),

        # Bipolar sigmoid: 2 * sigmoid(x) - 1
        'bipolar_sigmoid': BinaryOp('-',
            BinaryOp('*', c(2), _sigmoid()),
            c(1)),

        # Squared ReLU: relu(x)^2 = (0.5*(x+|x|))^2
        'squared_relu': BinaryOp('*',
            BinaryOp('*', c(0.5), BinaryOp('+', Var(), UnaryOp('abs', Var()))),
            BinaryOp('*', c(0.5), BinaryOp('+', Var(), UnaryOp('abs', Var())))),
    }
    return activations


def _tanh_of(inner_expr):
    """Build tanh(inner_expr) using expression tree: (exp(2*z) - 1) / (exp(2*z) + 1)."""
    exp2z = UnaryOp('exp', BinaryOp('*', Const(2), inner_expr))
    return BinaryOp('/', BinaryOp('-', exp2z, Const(1)), BinaryOp('+', exp2z, Const(1)))


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET LOADING (auto-download)
# ═══════════════════════════════════════════════════════════════════════════════
def load_dataset(config, device, data_root='./data'):
    """Load any torchvision dataset, auto-downloading if not present."""
    from torchvision import datasets, transforms

    name = config['name']
    loader_name = config['loader']
    do_flatten = config['flatten']

    log(f"  Loading {name}...")

    ds_class = getattr(datasets, loader_name)
    extra_kwargs = config.get('loader_kwargs', {})

    # Auto-download
    ds = ds_class(root=data_root, train=True, download=True,
                  transform=transforms.ToTensor(), **extra_kwargs)

    # Extract images
    if hasattr(ds, 'data'):
        if isinstance(ds.data, np.ndarray):
            images = torch.tensor(ds.data, dtype=torch.float32)
        else:
            images = ds.data.float()
    else:
        images = torch.stack([ds[i][0].squeeze() for i in range(len(ds))])

    # Extract labels
    if hasattr(ds, 'targets'):
        if isinstance(ds.targets, list):
            labels = torch.tensor(ds.targets, dtype=torch.long)
        else:
            labels = ds.targets.clone().detach().long()
    else:
        labels = torch.tensor([ds[i][1] for i in range(len(ds))], dtype=torch.long)

    if images.max() > 1.0:
        images = images.float() / 255.0
    if do_flatten:
        images = images.reshape(images.shape[0], -1)

    actual_dim = images.shape[1]
    config['input_dim'] = actual_dim

    # Subsample if needed
    max_samples = config.get('max_samples', None)
    if max_samples and len(images) > max_samples:
        torch.manual_seed(SEED + 999)
        perm = torch.randperm(len(images))[:max_samples]
        images = images[perm]
        labels = labels[perm]

    # Z-score normalize
    mu = images.mean(0, keepdim=True)
    std = images.std(0, keepdim=True).clamp(min=1e-6)
    images = (images - mu) / std

    # Shuffle
    torch.manual_seed(SEED)
    idx = torch.randperm(len(labels))
    images, labels = images[idx], labels[idx]

    # Split train/eval
    n = int(len(labels) * TRAIN_FRAC)
    train_x = images[:n].to(device)
    train_labels = labels[:n].to(device)
    eval_x = images[n:].to(device)
    eval_labels = labels[n:].to(device)

    n_classes = len(labels.unique())
    log(f"    {name}: {train_x.shape[0]} train, {eval_x.shape[0]} eval, "
        f"dim={actual_dim}, classes={n_classes}")
    return train_x, train_labels, eval_x, eval_labels, n_classes


# ═══════════════════════════════════════════════════════════════════════════════
# CHARACTERIZATION & FITNESS
# ═══════════════════════════════════════════════════════════════════════════════
def setup_characterization(input_dim, n_char_neurons, device):
    torch.manual_seed(SEED + 1000)
    weights = torch.randn(n_char_neurons, input_dim, device=device) * (2.0 / input_dim**0.5)
    bias = torch.zeros(n_char_neurons, device=device)
    return weights, bias


def characterize_activation(expr, train_x, train_labels, char_w, char_b, n_classes, device):
    n_samples = train_x.shape[0]
    accs = []

    for _ in range(N_EVAL_BATCHES):
        ix = torch.randint(0, n_samples, (EVAL_BATCH_SIZE,), device=device)
        batch = train_x[ix]
        labels = train_labels[ix]

        with torch.no_grad():
            linear = batch @ char_w.T + char_b
            activated = safe_eval_curve(expr, linear)

            if not torch.isfinite(activated).all():
                activated = torch.where(torch.isfinite(activated), activated,
                                       torch.zeros_like(activated))

            if activated.std() < 1e-6:
                return 10.0 / n_classes * 10, {'degenerate': True}

            half = EVAL_BATCH_SIZE // 2
            feat_norm = activated / (activated.norm(dim=-1, keepdim=True) + 1e-8)
            dists = torch.cdist(feat_norm[half:].unsqueeze(0),
                               feat_norm[:half].unsqueeze(0))[0]
            _, knn_idx = dists.topk(KNN_K, dim=-1, largest=False)
            knn_labels = labels[:half][knn_idx]
            class_ids = torch.arange(n_classes, device=device)
            votes = (knn_labels.unsqueeze(-1) == class_ids).float().sum(1)
            preds = votes.argmax(-1)
            acc = (preds == labels[half:]).float().mean().item() * 100
            accs.append(acc)

    mean_acc = np.mean(accs)
    stats = {
        'degenerate': False,
        'mean_acc': mean_acc,
        'std_acc': np.std(accs),
    }
    return mean_acc, stats


# ═══════════════════════════════════════════════════════════════════════════════
# EVOLUTION
# ═══════════════════════════════════════════════════════════════════════════════
def evolve_and_catalog(train_x, train_labels, char_w, char_b,
                       input_dim, n_classes, device, dataset_name,
                       pop_size=None, gens=None, n_random=None):
    """Evolve activation functions, return catalog list."""
    pop_size = pop_size or POP_SIZE
    gens = gens or GENS
    n_random = n_random or N_RANDOM_EXTRA

    catalog = []
    fingerprints = []
    expr_strings = set()
    score_cache = {}

    def add_to_catalog(expr, accuracy, stats, gen=-1):
        expr_str = str(expr)
        fp, raw_curve = curve_fingerprint(expr)
        if expr_str in expr_strings:
            return
        if fp is not None and is_duplicate(fp, fingerprints):
            return
        expr_strings.add(expr_str)
        fingerprints.append(fp)
        catalog.append({
            'id': len(catalog),
            'expression': expr_str,
            'accuracy': accuracy,
            'curve': raw_curve.tolist(),
            'depth': expr.depth(),
            'n_nodes': expr.size(),
            'generation': gen,
            'degenerate': stats.get('degenerate', False),
            'evolved_for': dataset_name,
        })

    def evaluate(expr):
        expr_str = str(expr)
        if expr_str in score_cache:
            return score_cache[expr_str], {'cached': True}
        acc, stats = characterize_activation(expr, train_x, train_labels,
                                             char_w, char_b, n_classes, device)
        score_cache[expr_str] = acc
        return acc, stats

    # Phase 0: Known activations (expanded set)
    log(f"    Seeding with known activations:")
    known = build_known_activations()
    for name, expr in known.items():
        acc, stats = evaluate(expr)
        add_to_catalog(expr, acc, stats, gen=-1)
        log(f"      {name:25s} -> {acc:.1f}%")

    # Phase 1: GP Evolution
    log(f"    GP Evolution ({pop_size} pop x {gens} gens)")
    population = []
    fitnesses = []
    for _ in range(pop_size):
        depth = random.randint(1, MAX_DEPTH)
        tree = random_tree(depth)
        while tree.size() > MAX_NODES:
            tree = random_tree(depth)
        population.append(tree)
        acc, stats = evaluate(tree)
        fitnesses.append(acc)
        add_to_catalog(tree, acc, stats, gen=0)

    best_acc = max(fitnesses)
    log(f"      Initial best: {best_acc:.1f}%, catalog={len(catalog)}")

    t0 = time.time()
    for gen in range(1, gens + 1):
        new_pop = []
        new_fit = []

        n_elite = max(1, int(pop_size * 0.05))
        ranked = sorted(range(pop_size), key=lambda i: fitnesses[i], reverse=True)
        for i in ranked[:n_elite]:
            new_pop.append(population[i].copy())
            new_fit.append(fitnesses[i])

        while len(new_pop) < pop_size:
            if random.random() < MUTATION_RATE:
                candidates = random.sample(range(pop_size), min(TOURNAMENT_SIZE, pop_size))
                parent_idx = max(candidates, key=lambda i: fitnesses[i])
                child = mutate(population[parent_idx])
            else:
                c1 = random.sample(range(pop_size), min(TOURNAMENT_SIZE, pop_size))
                c2 = random.sample(range(pop_size), min(TOURNAMENT_SIZE, pop_size))
                p1 = max(c1, key=lambda i: fitnesses[i])
                p2 = max(c2, key=lambda i: fitnesses[i])
                child = crossover(population[p1], population[p2])

            if child.size() > MAX_NODES or child.depth() > MAX_DEPTH + 1:
                child = random_tree(random.randint(1, MAX_DEPTH))

            acc, stats = evaluate(child)
            new_pop.append(child)
            new_fit.append(acc)
            add_to_catalog(child, acc, stats, gen=gen)

        population = new_pop
        fitnesses = new_fit

        cur_best = max(fitnesses)
        if cur_best > best_acc:
            best_acc = cur_best

        if gen % 25 == 0 or gen == gens:
            elapsed = time.time() - t0
            log(f"      gen {gen:>4}/{gens} | best={best_acc:.1f}% | "
                f"catalog={len(catalog)} | {elapsed:.0f}s")

    # Phase 2: Random exploration
    log(f"    Random exploration ({n_random} trees)")
    for i in range(n_random):
        depth = random.randint(1, MAX_DEPTH)
        tree = random_tree(depth)
        while tree.size() > MAX_NODES:
            tree = random_tree(depth)
        acc, stats = evaluate(tree)
        add_to_catalog(tree, acc, stats, gen=-2)

        if (i + 1) % 200 == 0:
            log(f"      {i+1}/{n_random} explored, catalog={len(catalog)}")

    n_good = sum(1 for c in catalog if c['accuracy'] > 100.0 / n_classes * 2)
    log(f"    CATALOG: {len(catalog)} unique | {n_good} above {100.0/n_classes*2:.0f}% | "
        f"best={max(c['accuracy'] for c in catalog):.1f}%")

    return catalog


# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-SCORING
# ═══════════════════════════════════════════════════════════════════════════════
def apply_activation_batch(linear, curves):
    """linear: [POP, B, N], curves: [POP, 200]. Returns [POP, B, N]."""
    P, B, N = linear.shape
    x = torch.clamp(linear, -4.99, 4.99)
    idx_float = (x + 5.0) / 10.0 * 199.0
    idx_low = idx_float.long()
    frac = idx_float - idx_low.float()
    p_idx = torch.arange(P, device=linear.device).view(P, 1, 1).expand(P, B, N)
    y_low = curves[p_idx, idx_low]
    y_high = curves[p_idx, torch.clamp(idx_low + 1, max=199)]
    return torch.clamp(y_low + frac * (y_high - y_low), -50, 50)


def knn_fitness(feat, labels, n_classes, k=KNN_K):
    """feat: [P, B, D]. Returns [P] accuracy %."""
    P, B, D = feat.shape
    half = B // 2
    feat_norm = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
    dists = torch.cdist(feat_norm[:, half:], feat_norm[:, :half])
    _, knn_idx = dists.topk(k, dim=-1, largest=False)
    knn_labels = labels[:half][knn_idx]
    class_ids = torch.arange(n_classes, device=feat.device)
    votes = (knn_labels.unsqueeze(-1) == class_ids).float().sum(2)
    preds = votes.argmax(-1)
    return (preds == labels[half:].unsqueeze(0)).float().mean(1) * 100.0


def cross_score_activations(all_activations, dataset_cache, device):
    """Score every activation on every dataset via k-NN classification.

    dataset_cache: dict of dataset_name -> (train_x, train_labels, char_w, char_b, n_classes)
    Returns list of dicts: [{task_key: score, ...}, ...]
    """
    n_total = len(all_activations)
    curves = torch.tensor([a['curve'] for a in all_activations],
                          dtype=torch.float32, device=device)

    cross_scores = [{} for _ in range(n_total)]
    chunk_size = 200

    for ds_name, (train_x, train_labels, char_w, char_b, n_classes) in dataset_cache.items():
        log(f"  Cross-scoring on {ds_name}...")
        n_samples = train_x.shape[0]
        t0 = time.time()

        for start in range(0, n_total, chunk_size):
            end = min(start + chunk_size, n_total)
            chunk_curves = curves[start:end]
            n_chunk = chunk_curves.shape[0]

            ix = torch.randint(0, n_samples, (EVAL_BATCH_SIZE,), device=device)
            batch = train_x[ix]
            labels = train_labels[ix]

            with torch.no_grad():
                linear = batch @ char_w.T + char_b
                lin_exp = linear.unsqueeze(0).expand(n_chunk, -1, -1)
                activated = apply_activation_batch(lin_exp, chunk_curves)
                scores = knn_fitness(activated, labels, n_classes)
                for i in range(n_chunk):
                    cross_scores[start + i][ds_name] = round(scores[i].item(), 2)

        elapsed = time.time() - t0
        log(f"    {ds_name}: scored {n_total} activations in {elapsed:.1f}s")

    return cross_scores


# ═══════════════════════════════════════════════════════════════════════════════
# COLLECT EXISTING CATALOGS
# ═══════════════════════════════════════════════════════════════════════════════
def collect_existing_catalogs():
    """Find and load all existing catalog.json files from results/."""
    results_dir = Path("results")
    if not results_dir.exists():
        return []

    all_acts = []
    found = []

    # Primordial catalogs
    for p in sorted(results_dir.glob("primordial_*/catalog.json")):
        found.append(('MNIST_primordial', p))

    # Experiment catalogs
    exp_dirs = sorted(d for d in results_dir.glob("experiment_*") if d.is_dir())
    if exp_dirs:
        latest = exp_dirs[-1]
        for ds_dir in latest.iterdir():
            if ds_dir.is_dir() and (ds_dir / 'catalog.json').exists():
                found.append((ds_dir.name, ds_dir / 'catalog.json'))

    # Combined catalogs
    for p in sorted(results_dir.glob("combined_catalog_*/combined_catalog.json")):
        found.append(('combined', p))

    # Master catalogs
    for p in sorted(results_dir.glob("master_catalog_*/master_catalog.json")):
        found.append(('master', p))

    for name, path in found:
        log(f"  Found catalog: {name} -> {path}")
        try:
            with open(path) as f:
                data = json.load(f)
            acts = data.get('activations', [])
            non_deg = [a for a in acts if not a.get('degenerate', False)]
            for a in non_deg:
                a.setdefault('evolved_for', name)
                a.setdefault('scores', {})
            all_acts.extend(non_deg)
            log(f"    Loaded {len(non_deg)} non-degenerate activations")
        except Exception as e:
            log(f"    ERROR loading {path}: {e}")

    return all_acts


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED: compute t-SNE layout + category data (reused by all renderers)
# ═══════════════════════════════════════════════════════════════════════════════
def _prepare_layout(all_activations, task_keys):
    """Compute t-SNE 2D + 3D positions from CURVE fingerprints, scores for coloring."""
    n = len(all_activations)

    # Scores are used for coloring, not for spatial layout
    scores = np.zeros((n, len(task_keys)), dtype=np.float32)
    for i, a in enumerate(all_activations):
        for j, key in enumerate(task_keys):
            scores[i, j] = a.get('scores', {}).get(key, 0.0)
    np.nan_to_num(scores, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    mean_scores = scores.mean(axis=1)

    # Use 200-point curve fingerprints for spatial embedding — this captures
    # the mathematical shape of each activation, so similar functions cluster.
    curves = np.zeros((n, N_CURVE_POINTS), dtype=np.float32)
    for i, a in enumerate(all_activations):
        c = a.get('curve', [])
        if len(c) == N_CURVE_POINTS:
            curves[i] = c
        elif len(c) > 0:
            # Resample to N_CURVE_POINTS if different length
            arr = np.array(c, dtype=np.float32)
            x_old = np.linspace(0, 1, len(arr))
            x_new = np.linspace(0, 1, N_CURVE_POINTS)
            curves[i] = np.interp(x_new, x_old, arr)

    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    # PCA to reduce 200-dim curves to 50 dims before t-SNE (faster + stable)
    n_pca = min(50, n - 1, N_CURVE_POINTS)
    pca = PCA(n_components=n_pca)
    curves_reduced = pca.fit_transform(curves)
    log(f"  PCA: {N_CURVE_POINTS} curve dims -> {n_pca} dims "
        f"({pca.explained_variance_ratio_.sum():.1%} variance)")

    perplexity = min(30, max(5, n // 5))

    log(f"  Computing t-SNE 2D layout ({n} points)...")
    tsne2 = TSNE(n_components=2, perplexity=perplexity, max_iter=1000, random_state=42)
    pos_2d = tsne2.fit_transform(curves_reduced).astype(np.float32)
    for dim in range(2):
        lo, hi = pos_2d[:, dim].min(), pos_2d[:, dim].max()
        pos_2d[:, dim] = (pos_2d[:, dim] - lo) / (hi - lo + 1e-10)

    log(f"  Computing t-SNE 3D layout ({n} points)...")
    tsne3 = TSNE(n_components=3, perplexity=perplexity, max_iter=1000, random_state=42)
    pos_3d = tsne3.fit_transform(curves_reduced).astype(np.float32)
    for dim in range(3):
        lo, hi = pos_3d[:, dim].min(), pos_3d[:, dim].max()
        pos_3d[:, dim] = (pos_3d[:, dim] - lo) / (hi - lo + 1e-10)

    categories = [a.get('evolved_for', 'unknown') for a in all_activations]
    unique_cats = sorted(set(categories))
    cat_to_idx = {cat: i for i, cat in enumerate(unique_cats)}

    palette = [
        '#64B4FF', '#FF7F64', '#B478FF', '#64E6B4',
        '#FFD250', '#50DCDC', '#468CC8', '#C86450',
        '#8C50C8', '#50B48C', '#C8AA3C', '#3CB4B4',
    ]

    return {
        'n': n, 'scores': scores, 'mean_scores': mean_scores,
        'pos_2d': pos_2d, 'pos_3d': pos_3d,
        'categories': categories, 'unique_cats': unique_cats,
        'cat_to_idx': cat_to_idx, 'palette': palette,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PNG 1: Main galaxy overview (category-colored)
# ═══════════════════════════════════════════════════════════════════════════════
def render_galaxy_overview(all_activations, layout, out_dir):
    """Full galaxy scatter colored by category with top-generalist labels."""
    pos_2d = layout['pos_2d']
    mean_scores = layout['mean_scores']
    categories = layout['categories']
    unique_cats = layout['unique_cats']
    cat_to_idx = layout['cat_to_idx']
    palette = layout['palette']
    n = layout['n']

    score_norm = (mean_scores - mean_scores.min()) / (mean_scores.max() - mean_scores.min() + 1e-10)
    sizes = 3 + 40 * score_norm ** 2
    top_10 = np.argsort(mean_scores)[-10:][::-1]

    fig, ax = plt.subplots(figsize=(16, 14), facecolor='#06080E')
    ax.set_facecolor('#06080E')

    for cat_name in unique_cats:
        mask = np.array([c == cat_name for c in categories])
        color = palette[cat_to_idx[cat_name] % len(palette)]
        ax.scatter(pos_2d[mask, 0], pos_2d[mask, 1],
                   s=sizes[mask], c=color, alpha=0.6,
                   edgecolors='none', label=cat_name, rasterized=True)

    ax.scatter(pos_2d[top_10, 0], pos_2d[top_10, 1],
               s=250, c='#FFE040', marker='*', edgecolors='white',
               linewidths=0.5, zorder=10, label='Top 10 Generalists')

    for rank, idx in enumerate(top_10[:7]):
        expr = all_activations[idx].get('expression', '?')
        if len(expr) > 35:
            expr = expr[:32] + '...'
        ax.annotate(
            f"#{rank+1}: {expr}\n(mean={mean_scores[idx]:.1f})",
            xy=(pos_2d[idx, 0], pos_2d[idx, 1]),
            xytext=(12, 12 + (rank % 3) * 8), textcoords='offset points',
            fontsize=7, color='#FFE040',
            arrowprops=dict(arrowstyle='->', color='#FFE040', lw=0.5),
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#101420',
                      edgecolor='#FFE040', alpha=0.85))

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('t-SNE dim 1', color='#8090A0', fontsize=11)
    ax.set_ylabel('t-SNE dim 2', color='#8090A0', fontsize=11)
    ax.set_title(f'Activation Galaxy — {n:,} activations (category coloring)',
                 color='#C8DCF0', fontsize=16, fontweight='bold', pad=14)
    ax.tick_params(colors='#506070', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#2A3040')

    legend = ax.legend(loc='lower left', fontsize=8, framealpha=0.8,
                       facecolor='#101420', edgecolor='#3C5078', labelcolor='#C0C8D8')
    legend.get_frame().set_linewidth(0.5)

    path = str(out_dir / "galaxy_overview.png")
    fig.savefig(path, dpi=150, facecolor='#06080E', bbox_inches='tight')
    plt.close(fig)
    log(f"  Saved {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# PNG 2-N: Per-task heatmaps (one image each)
# ═══════════════════════════════════════════════════════════════════════════════
def render_task_heatmaps(all_activations, layout, task_keys, out_dir):
    """One PNG per task showing score heatmap on the 2D layout."""
    pos_2d = layout['pos_2d']
    n = layout['n']
    paths = []

    for tk in task_keys:
        task_scores = np.array([a.get('scores', {}).get(tk, 0.0)
                                for a in all_activations], dtype=np.float32)
        lo, hi = task_scores.min(), task_scores.max()
        t_norm = (task_scores - lo) / (hi - lo + 1e-10)
        sizes = 2 + 25 * t_norm ** 2

        top5 = np.argsort(task_scores)[-5:][::-1]

        fig, ax = plt.subplots(figsize=(14, 12), facecolor='#06080E')
        ax.set_facecolor('#06080E')

        sc = ax.scatter(pos_2d[:, 0], pos_2d[:, 1],
                        s=sizes, c=task_scores, cmap='RdYlBu_r',
                        vmin=lo, vmax=hi, alpha=0.7,
                        edgecolors='none', rasterized=True)
        cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label('Score', color='#8090A0', fontsize=10)
        cbar.ax.tick_params(colors='#8090A0', labelsize=8)

        ax.scatter(pos_2d[top5, 0], pos_2d[top5, 1],
                   s=180, facecolors='none', edgecolors='#FFE040',
                   linewidths=1.5, zorder=10)

        for rank, idx in enumerate(top5):
            expr = all_activations[idx].get('expression', '?')
            if len(expr) > 30:
                expr = expr[:27] + '...'
            ax.annotate(
                f"#{rank+1}: {task_scores[idx]:.1f}  {expr}",
                xy=(pos_2d[idx, 0], pos_2d[idx, 1]),
                xytext=(10, 8 + (rank % 3) * 6), textcoords='offset points',
                fontsize=7, color='#FFE040',
                arrowprops=dict(arrowstyle='->', color='#FFE040', lw=0.5),
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#101420',
                          edgecolor='#FFE040', alpha=0.85))

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('t-SNE dim 1', color='#8090A0', fontsize=10)
        ax.set_ylabel('t-SNE dim 2', color='#8090A0', fontsize=10)
        ax.set_title(f'Heatmap: {tk}  (range {lo:.1f} – {hi:.1f})',
                     color='#C8DCF0', fontsize=14, fontweight='bold', pad=12)
        ax.tick_params(colors='#506070', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#2A3040')

        safe_name = tk.replace('/', '_')
        path = str(out_dir / f"heatmap_{safe_name}.png")
        fig.savefig(path, dpi=150, facecolor='#06080E', bbox_inches='tight')
        plt.close(fig)
        paths.append(path)
        log(f"  Saved {path}")

    return paths


# ═══════════════════════════════════════════════════════════════════════════════
# PNG: Top activations detail sheet
# ═══════════════════════════════════════════════════════════════════════════════
def render_top_activations(all_activations, layout, task_keys, out_dir):
    """Grid of top-20 generalist activation curves + score radar."""
    mean_scores = layout['mean_scores']
    top_20 = np.argsort(mean_scores)[-20:][::-1]

    n_cols, n_rows = 5, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 14), facecolor='#06080E')
    fig.suptitle('Top 20 Generalist Activations', fontsize=18,
                 color='#C8DCF0', fontweight='bold', y=0.98)

    xvals = np.linspace(-5, 5, 200)

    for rank, idx in enumerate(top_20):
        row, col = divmod(rank, n_cols)
        ax = axes[row, col]
        ax.set_facecolor('#10141C')

        curve = np.array(all_activations[idx].get('curve', [0]*200)[:200])
        curve = np.clip(curve, -20, 20)
        ax.plot(xvals, curve, color='#A0D8FF', linewidth=1.5)
        if curve.min() < 0 < curve.max():
            ax.axhline(y=0, color='#3C4860', linewidth=0.4)

        expr = all_activations[idx].get('expression', '?')
        if len(expr) > 35:
            expr = expr[:32] + '...'
        score_parts = []
        for tk in task_keys:
            s = all_activations[idx].get('scores', {}).get(tk, 0)
            short = tk[:6]
            score_parts.append(f"{short}={s:.0f}")
        score_str = '  '.join(score_parts)

        ax.set_title(f"#{rank+1}  mean={mean_scores[idx]:.1f}",
                     fontsize=8, color='#FFE040', pad=4)
        ax.text(0.5, -0.02, expr, transform=ax.transAxes, fontsize=6,
                color='#90C0A0', ha='center', va='top', fontfamily='monospace')
        ax.text(0.5, -0.10, score_str, transform=ax.transAxes, fontsize=5,
                color='#7090A0', ha='center', va='top', fontfamily='monospace')

        ax.tick_params(colors='#506070', labelsize=6)
        for spine in ax.spines.values():
            spine.set_color('#3C5078')

    fig.subplots_adjust(hspace=0.55, wspace=0.3)
    path = str(out_dir / "top_activations.png")
    fig.savefig(path, dpi=150, facecolor='#06080E', bbox_inches='tight')
    plt.close(fig)
    log(f"  Saved {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# HTML: Interactive 3D galaxy (plotly, self-contained)
# ═══════════════════════════════════════════════════════════════════════════════
def render_galaxy_3d_html(all_activations, layout, task_keys, out_dir):
    """Render an interactive 3D t-SNE galaxy as a self-contained HTML file."""
    import plotly.graph_objects as go

    pos_3d = layout['pos_3d']
    mean_scores = layout['mean_scores']
    categories = layout['categories']
    unique_cats = layout['unique_cats']
    cat_to_idx = layout['cat_to_idx']
    palette = layout['palette']
    n = layout['n']

    score_norm = (mean_scores - mean_scores.min()) / (mean_scores.max() - mean_scores.min() + 1e-10)
    sizes = 2 + 6 * score_norm

    top_10 = np.argsort(mean_scores)[-10:][::-1]

    fig = go.Figure()

    # One trace per category for legend toggling
    for cat_name in unique_cats:
        mask = np.array([c == cat_name for c in categories])
        idxs = np.where(mask)[0]
        color = palette[cat_to_idx[cat_name] % len(palette)]

        hover_texts = []
        for i in idxs:
            expr = all_activations[i].get('expression', '?')
            if len(expr) > 60:
                expr = expr[:57] + '...'
            parts = [f"<b>#{all_activations[i].get('id', i)}</b>  {cat_name}",
                     f"<b>Expression:</b> {expr}",
                     f"<b>Mean score:</b> {mean_scores[i]:.1f}"]
            for tk in task_keys:
                s = all_activations[i].get('scores', {}).get(tk, 0)
                parts.append(f"  {tk}: {s:.1f}")
            hover_texts.append('<br>'.join(parts))

        fig.add_trace(go.Scatter3d(
            x=pos_3d[idxs, 0], y=pos_3d[idxs, 1], z=pos_3d[idxs, 2],
            mode='markers',
            marker=dict(size=sizes[idxs], color=color, opacity=0.6,
                        line=dict(width=0)),
            name=cat_name,
            text=hover_texts,
            hoverinfo='text',
        ))

    # Top-10 generalist stars as separate trace
    top_hover = []
    for rank, idx in enumerate(top_10):
        expr = all_activations[idx].get('expression', '?')
        if len(expr) > 60:
            expr = expr[:57] + '...'
        parts = [f"<b>★ #{rank+1} GENERALIST</b>",
                 f"<b>Expression:</b> {expr}",
                 f"<b>Mean score:</b> {mean_scores[idx]:.1f}"]
        for tk in task_keys:
            s = all_activations[idx].get('scores', {}).get(tk, 0)
            parts.append(f"  {tk}: {s:.1f}")
        top_hover.append('<br>'.join(parts))

    fig.add_trace(go.Scatter3d(
        x=pos_3d[top_10, 0], y=pos_3d[top_10, 1], z=pos_3d[top_10, 2],
        mode='markers+text',
        marker=dict(size=8, color='#FFE040', opacity=1.0,
                    symbol='diamond',
                    line=dict(width=1, color='white')),
        text=[f"#{r+1}" for r in range(len(top_10))],
        textposition='top center',
        textfont=dict(size=9, color='#FFE040'),
        name='Top 10 Generalists',
        hovertext=top_hover,
        hoverinfo='text',
    ))

    # Per-task heatmap buttons
    buttons = []
    # "Category" button (default)
    buttons.append(dict(
        label='Category Colors',
        method='update',
        args=[{'visible': [True] * (len(unique_cats) + 1)}]
    ))

    # We'll add invisible heatmap traces, one per task, then toggle visibility
    n_base_traces = len(unique_cats) + 1  # category traces + top10

    for ti, tk in enumerate(task_keys):
        task_scores = np.array([a.get('scores', {}).get(tk, 0.0)
                                for a in all_activations], dtype=np.float32)
        lo, hi = task_scores.min(), task_scores.max()

        hover_texts = []
        for i in range(n):
            expr = all_activations[i].get('expression', '?')
            if len(expr) > 50:
                expr = expr[:47] + '...'
            hover_texts.append(
                f"<b>{tk}: {task_scores[i]:.1f}</b><br>"
                f"{expr}<br>Mean: {mean_scores[i]:.1f}")

        fig.add_trace(go.Scatter3d(
            x=pos_3d[:, 0], y=pos_3d[:, 1], z=pos_3d[:, 2],
            mode='markers',
            marker=dict(size=sizes, color=task_scores,
                        colorscale='RdYlBu_r', cmin=lo, cmax=hi,
                        opacity=0.6, colorbar=dict(
                            title=dict(text=tk[:15],
                                       font=dict(color='#C8DCF0', size=11)),
                            len=0.5, thickness=15,
                            x=1.02, xpad=5,
                            tickfont=dict(color='#AAB0C0', size=10),
                        ),
                        line=dict(width=0)),
            name=f'Heatmap: {tk}',
            text=hover_texts,
            hoverinfo='text',
            visible=False,
        ))

        # Build visibility: hide all category traces, show this heatmap + top10
        vis = [False] * n_base_traces + [False] * len(task_keys)
        vis[len(unique_cats)] = True   # top10 always visible
        vis[n_base_traces + ti] = True
        buttons.append(dict(label=tk[:15], method='update', args=[{'visible': vis}]))

    # Default visibility: category traces + top10 visible, heatmaps hidden
    default_vis = [True] * n_base_traces + [False] * len(task_keys)
    fig.for_each_trace(lambda t: t.update(visible=True)
                       if t.name in unique_cats or t.name == 'Top 10 Generalists'
                       else t.update(visible=False))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#06080E',
        plot_bgcolor='#06080E',
        title=dict(
            text=f'Activation Galaxy — {n:,} activations (3D t-SNE)',
            font=dict(size=20, color='#C8DCF0'),
            x=0.5,
        ),
        scene=dict(
            xaxis=dict(title='t-SNE 1', gridcolor='#1A2030', zerolinecolor='#1A2030',
                       backgroundcolor='#06080E', color='#506070'),
            yaxis=dict(title='t-SNE 2', gridcolor='#1A2030', zerolinecolor='#1A2030',
                       backgroundcolor='#06080E', color='#506070'),
            zaxis=dict(title='t-SNE 3', gridcolor='#1A2030', zerolinecolor='#1A2030',
                       backgroundcolor='#06080E', color='#506070'),
            bgcolor='#06080E',
        ),
        legend=dict(
            bgcolor='rgba(16,20,28,0.85)', bordercolor='#3C5078', borderwidth=1,
            font=dict(size=11, color='#C0C8D8'),
            itemsizing='constant',
        ),
        updatemenus=[dict(
            type='dropdown', direction='down',
            x=0.01, y=0.99, xanchor='left', yanchor='top',
            bgcolor='#10141C', bordercolor='#3C5078',
            font=dict(color='#C0C8D8', size=11),
            buttons=buttons,
            active=0,
        )],
        margin=dict(l=0, r=0, t=50, b=0),
        width=1400,
        height=900,
    )

    path = str(out_dir / "galaxy_3d.html")
    fig.write_html(path, include_plotlyjs=True, full_html=True)
    log(f"  Saved {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER RENDER: calls all renderers, returns dict of paths
# ═══════════════════════════════════════════════════════════════════════════════
def render_all(all_activations, task_keys, out_dir):
    """Render separate PNGs + interactive 3D HTML. Returns dict of output paths."""
    out_dir = Path(out_dir)
    n = len(all_activations)
    log(f"  Rendering galaxy with {n} activations...")

    layout = _prepare_layout(all_activations, task_keys)

    paths = {}
    paths['overview'] = render_galaxy_overview(all_activations, layout, out_dir)
    paths['heatmaps'] = render_task_heatmaps(all_activations, layout, task_keys, out_dir)
    paths['top_activations'] = render_top_activations(all_activations, layout, task_keys, out_dir)
    paths['galaxy_3d'] = render_galaxy_3d_html(all_activations, layout, task_keys, out_dir)

    return paths


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    global _LOG_FILE

    # Parse args
    quick_mode = '--quick' in sys.argv
    catalog_only = None
    for arg in sys.argv[1:]:
        if arg.startswith('--catalog='):
            catalog_only = arg.split('=', 1)[1]
        elif arg == '--catalog' and sys.argv.index(arg) + 1 < len(sys.argv):
            catalog_only = sys.argv[sys.argv.index(arg) + 1]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.set_float32_matmul_precision("high")

    sid = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path("results") / f"galaxy_{sid}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    _LOG_FILE = str(exp_dir / "galaxy_log.txt")

    log("=" * 70)
    log("ACTIVATION GALAXY EXPLORER — FULL PIPELINE")
    log("=" * 70)
    log(f"  Device: {device}")
    if torch.cuda.is_available():
        log(f"  GPU: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        log(f"  GPU memory: {mem:.1f} GB")
    log(f"  Quick mode: {quick_mode}")
    log(f"  Output: {exp_dir}")
    log("")

    t_total = time.time()

    # Evolution params
    pop = QUICK_POP_SIZE if quick_mode else POP_SIZE
    gen = QUICK_GENS if quick_mode else GENS
    n_rand = QUICK_RANDOM_EXTRA if quick_mode else N_RANDOM_EXTRA

    # ══════════════════════════════════════════════════════════════════════
    # If --catalog provided, skip to rendering
    # ══════════════════════════════════════════════════════════════════════
    if catalog_only:
        log(f"CATALOG-ONLY MODE: Loading {catalog_only}")
        with open(catalog_only) as f:
            data = json.load(f)
        all_activations = data.get('activations', [])
        task_keys = data.get('task_grid', TASK_KEYS)
        log(f"  Loaded {len(all_activations)} activations")

        paths = render_all(all_activations, task_keys, exp_dir)

        log(f"\nDONE. Outputs:")
        for name, p in paths.items():
            if isinstance(p, list):
                for pp in p:
                    log(f"  {pp}")
            else:
                log(f"  {p}")
        return

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 1: Download datasets & load
    # ══════════════════════════════════════════════════════════════════════
    log("PHASE 1: Loading datasets (auto-downloading if needed)")
    log("-" * 50)

    dataset_cache = {}  # name -> (train_x, train_labels, eval_x, eval_labels, n_classes)
    char_cache = {}     # name -> (char_w, char_b)

    for config in DATASET_CONFIGS:
        ds_name = config['name']
        try:
            train_x, train_labels, eval_x, eval_labels, n_classes = \
                load_dataset(config, device, data_root='./data')
            dataset_cache[ds_name] = (train_x, train_labels, eval_x, eval_labels, n_classes)

            n_char = config.get('n_char_neurons', N_CHAR_NEURONS)
            input_dim = config['input_dim']
            char_w, char_b = setup_characterization(input_dim, n_char, device)
            char_cache[ds_name] = (char_w, char_b)
        except Exception as e:
            log(f"  ERROR loading {ds_name}: {e}")
            import traceback
            traceback.print_exc()

    log(f"  Loaded {len(dataset_cache)} datasets: {list(dataset_cache.keys())}")
    log("")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 2: Collect existing catalogs
    # ══════════════════════════════════════════════════════════════════════
    log("PHASE 2: Collecting existing catalogs")
    log("-" * 50)
    existing = collect_existing_catalogs()
    log(f"  Total existing activations: {len(existing)}")
    log("")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 3: Evolve new activations per dataset
    # ══════════════════════════════════════════════════════════════════════
    log("PHASE 3: Evolving activations per dataset")
    log("-" * 50)

    per_dataset_catalogs = {}

    for config in DATASET_CONFIGS:
        ds_name = config['name']
        if ds_name not in dataset_cache:
            log(f"  Skipping {ds_name} (not loaded)")
            continue

        train_x, train_labels, eval_x, eval_labels, n_classes = dataset_cache[ds_name]
        char_w, char_b = char_cache[ds_name]
        input_dim = config['input_dim']

        log(f"\n  === Evolving for {ds_name} (dim={input_dim}, classes={n_classes}) ===")

        catalog = evolve_and_catalog(
            train_x, train_labels, char_w, char_b,
            input_dim, n_classes, device, ds_name,
            pop_size=pop, gens=gen, n_random=n_rand)

        per_dataset_catalogs[ds_name] = catalog

        # Save per-dataset catalog
        cat_path = exp_dir / f"catalog_{ds_name}.json"
        with open(cat_path, 'w', encoding='utf-8') as f:
            json.dump({
                'dataset': ds_name,
                'n_activations': len(catalog),
                'activations': catalog,
            }, f, indent=2, default=str)
        log(f"    Saved {cat_path.name} ({len(catalog)} activations)")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log(f"\n  Phase 3 complete. Per-dataset catalog sizes:")
    for ds_name, cat in per_dataset_catalogs.items():
        log(f"    {ds_name}: {len(cat)}")
    log("")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 4: Merge & deduplicate all catalogs
    # ══════════════════════════════════════════════════════════════════════
    log("PHASE 4: Merging and deduplicating all catalogs")
    log("-" * 50)

    all_merged = list(existing)
    for ds_name, cat in per_dataset_catalogs.items():
        all_merged.extend(cat)

    log(f"  Total before dedup: {len(all_merged)}")
    t_dedup = time.time()
    all_deduped = deduplicate_by_curve(all_merged, threshold=0.999)
    dedup_time = time.time() - t_dedup
    removed = len(all_merged) - len(all_deduped)
    log(f"  Removed {removed} duplicates in {dedup_time:.1f}s")
    log(f"  Total after dedup: {len(all_deduped)}")

    # Re-index
    for i, a in enumerate(all_deduped):
        a['id'] = i
    log("")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 5: Cross-score all activations on all datasets
    # ══════════════════════════════════════════════════════════════════════
    log("PHASE 5: Cross-scoring all activations on all datasets")
    log("-" * 50)

    # Build scoring cache
    scoring_cache = {}
    for ds_name in dataset_cache:
        train_x, train_labels, _, _, n_classes = dataset_cache[ds_name]
        char_w, char_b = char_cache[ds_name]
        scoring_cache[ds_name] = (train_x, train_labels, char_w, char_b, n_classes)

    cross_scores = cross_score_activations(all_deduped, scoring_cache, device)

    # Attach scores
    active_task_keys = list(scoring_cache.keys())
    for i, a in enumerate(all_deduped):
        a['scores'] = cross_scores[i]

    log("")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 6: Run basis expansion tests on all datasets
    # ══════════════════════════════════════════════════════════════════════
    log("PHASE 6: Running basis expansion tests on all datasets")
    log("-" * 50)

    # Import basis expansion from dataset_experiment if available
    test_results = {}
    try:
        from dataset_experiment import (
            run_basis_expansion, load_catalog_from_data, BASIS_N_NEURONS, BASIS_MAX_ELEMENTS
        )

        for config in DATASET_CONFIGS:
            ds_name = config['name']
            if ds_name not in dataset_cache:
                continue

            train_x, train_labels, eval_x, eval_labels, n_classes = dataset_cache[ds_name]
            input_dim = config['input_dim']
            n_neurons = config.get('n_basis_neurons', BASIS_N_NEURONS)
            max_elements = config.get('max_elements', BASIS_MAX_ELEMENTS)

            log(f"\n  Testing basis expansion on {ds_name}...")
            try:
                results, checkpoint, _, _ = run_basis_expansion(
                    all_deduped, train_x, train_labels, eval_x, eval_labels,
                    input_dim, n_classes, n_neurons, device, max_elements=max_elements)

                test_results[ds_name] = {
                    'heldout_accuracy': results.get('heldout_accuracy', 0),
                    'train_accuracy': results.get('train_accuracy', 0),
                    'n_elements': results.get('n_elements', 0),
                    'total_dim': results.get('total_dim', 0),
                }

                if checkpoint:
                    torch.save(checkpoint, exp_dir / f'model_{ds_name}.pt')

                with open(exp_dir / f'basis_results_{ds_name}.json', 'w') as f:
                    json.dump(results, f, indent=2, default=str)

                log(f"    {ds_name}: held-out={results.get('heldout_accuracy', 0):.1f}%, "
                    f"train={results.get('train_accuracy', 0):.1f}%, "
                    f"dim={results.get('total_dim', 0)}")
            except Exception as e:
                log(f"    ERROR in basis expansion for {ds_name}: {e}")
                import traceback
                traceback.print_exc()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except ImportError:
        log("  dataset_experiment.py not found, skipping basis expansion tests")

    log("")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 7: Save master catalog + render galaxy PNG
    # ══════════════════════════════════════════════════════════════════════
    log("PHASE 7: Saving master catalog and rendering galaxy")
    log("-" * 50)

    master_path = exp_dir / "master_catalog.json"
    with open(master_path, 'w', encoding='utf-8') as f:
        json.dump({
            'version': 'galaxy_explorer_v2',
            'timestamp': sid,
            'n_activations': len(all_deduped),
            'task_grid': active_task_keys,
            'activations': all_deduped,
        }, f, indent=2, default=str)
    log(f"  Saved master_catalog.json ({len(all_deduped)} activations)")

    # Free GPU memory before rendering
    del dataset_cache, char_cache, scoring_cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Render galaxy (separate PNGs + 3D HTML)
    paths = render_all(all_deduped, active_task_keys, exp_dir)

    total_time = time.time() - t_total

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    log(f"\n{'=' * 80}")
    log("GALAXY EXPLORER SUMMARY")
    log(f"{'=' * 80}")
    log(f"  Total activations: {len(all_deduped)}")
    log(f"  Existing catalogs collected: {len(existing)}")
    for ds_name, cat in per_dataset_catalogs.items():
        log(f"  Evolved for {ds_name}: {len(cat)}")
    log(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    log("")

    # Basis expansion results
    if test_results:
        log("  BASIS EXPANSION TEST RESULTS:")
        log(f"  {'Dataset':<15} | {'Dim':>5} | {'Train':>7} | {'Held-out':>8}")
        log(f"  {'-' * 45}")
        for ds_name, r in test_results.items():
            log(f"  {ds_name:<15} | {r['total_dim']:>5} | "
                f"{r['train_accuracy']:>6.1f}% | {r['heldout_accuracy']:>7.1f}%")
        log("")

    # Top generalists
    mean_scores = np.array([
        np.mean([v for v in a.get('scores', {}).values() if v > -99]) if a.get('scores') else 0
        for a in all_deduped
    ])
    top_10 = np.argsort(mean_scores)[-10:][::-1]
    log("  TOP 10 GENERALIST ACTIVATIONS:")
    for rank, idx in enumerate(top_10):
        ms = mean_scores[idx]
        expr = all_deduped[idx].get('expression', '?')
        if len(expr) > 50:
            expr = expr[:47] + '...'
        log(f"    #{rank+1}: mean={ms:>5.1f}  {expr}")

    log(f"\n  Output directory: {exp_dir}")
    log(f"  Outputs:")
    for name, p in paths.items():
        if isinstance(p, list):
            for pp in p:
                log(f"    {pp}")
        else:
            log(f"    {p}")
    log("DONE.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\nInterrupted by user.")
    except Exception as e:
        log(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
