"""
Evolve activation functions from PRIMORDIAL mathematical operations.
====================================================================
No named activations. No sin, relu, tanh as building blocks.
The building blocks are the true atoms of computation:

  Terminals: x, 0, 0.5, 1, -1, 2, e, π
  Unary ops: neg, abs, exp, log, sin, cos
  Binary ops: +, -, *, /

Everything else EMERGES from these: relu, sigmoid, tanh, gauss, swish,
and thousands of unnamed activation shapes that evolution discovers.

MNIST is the characterization function — every activation gets scored
by how well it separates digits when used as a neural activation.
We record EVERYTHING. Every unique activation shape, good or bad.

This builds the "periodic table of calculations."

Usage: python evolve_primitives.py
"""

import torch
import numpy as np
import math
import random
import json
import copy
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ================================================================
# CONFIG
# ================================================================
SEED = 42
INPUT_DIM = 784
N_CLASSES = 10
TRAIN_FRAC = 0.8

# Characterization: fixed random projections
N_CHAR_NEURONS = 16     # neurons for characterization
N_EVAL_BATCHES = 5      # batches to average for stable score
EVAL_BATCH_SIZE = 2000
KNN_K = 7

# Evolution
POP_SIZE = 300
GENS = 200
MAX_DEPTH = 5
MAX_NODES = 25
TOURNAMENT_SIZE = 5
MUTATION_RATE = 0.7     # probability of mutation vs crossover
SUBTREE_MUT_RATE = 0.4  # of mutations: subtree replacement
POINT_MUT_RATE = 0.3    # of mutations: change op
CONST_MUT_RATE = 0.2    # of mutations: tweak constant
HOIST_MUT_RATE = 0.1    # of mutations: shrink tree

# Additional random exploration
N_RANDOM_EXTRA = 1000

# ================================================================
# PRIMORDIAL OPERATIONS
# ================================================================
UNARY_OPS = ['neg', 'abs', 'exp', 'log', 'sin', 'cos']
BINARY_OPS = ['+', '-', '*', '/']
CONSTANTS = [0.0, 0.5, 1.0, -1.0, 2.0, math.e, math.pi]


# ================================================================
# EXPRESSION TREE
# ================================================================
class Expr:
    """Base class for expression tree nodes."""
    def eval(self, x):
        raise NotImplementedError
    def depth(self):
        raise NotImplementedError
    def size(self):
        raise NotImplementedError
    def copy(self):
        return copy.deepcopy(self)
    def nodes(self):
        """Return list of all nodes in the tree (for subtree operations)."""
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
            return 'π'
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
            # Safe division
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


# ================================================================
# RANDOM TREE GENERATION
# ================================================================
def random_terminal():
    if random.random() < 0.55:
        return Var()
    else:
        return Const(random.choice(CONSTANTS))


def random_tree(max_depth, current_depth=0):
    """Generate random expression tree. Biased toward shorter trees."""
    # Force terminal at max depth
    if current_depth >= max_depth:
        return random_terminal()

    # Increasing probability of terminal as we go deeper
    p_terminal = 0.2 + 0.15 * current_depth
    if current_depth > 0 and random.random() < p_terminal:
        return random_terminal()

    # Choose unary or binary
    if random.random() < 0.45:
        op = random.choice(UNARY_OPS)
        child = random_tree(max_depth, current_depth + 1)
        return UnaryOp(op, child)
    else:
        op = random.choice(BINARY_OPS)
        left = random_tree(max_depth, current_depth + 1)
        right = random_tree(max_depth, current_depth + 1)
        return BinaryOp(op, left, right)


# ================================================================
# GENETIC OPERATORS
# ================================================================
def get_random_subtree(tree):
    """Return a random node from the tree."""
    all_nodes = tree.nodes()
    return random.choice(all_nodes)


def replace_random_subtree(tree, new_subtree):
    """Replace a random node in tree with new_subtree. Returns new tree."""
    tree = tree.copy()
    all_nodes = tree.nodes()
    if len(all_nodes) <= 1:
        return new_subtree.copy()

    # Pick a non-root node to replace
    target_idx = random.randint(1, len(all_nodes) - 1)
    target = all_nodes[target_idx]

    # Find parent and replace
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
    """Apply a random mutation to the tree."""
    r = random.random()

    if r < SUBTREE_MUT_RATE:
        # Subtree replacement
        new_sub = random_tree(max_depth=3, current_depth=0)
        return replace_random_subtree(tree, new_sub)

    elif r < SUBTREE_MUT_RATE + POINT_MUT_RATE:
        # Point mutation: change an op
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
        # Constant mutation: tweak a constant
        tree = tree.copy()
        all_nodes = tree.nodes()
        consts = [n for n in all_nodes if isinstance(n, Const)]
        if consts:
            node = random.choice(consts)
            node.value += random.gauss(0, 0.5)
        else:
            # No constants, swap a variable for a constant
            vars_ = [n for n in all_nodes if isinstance(n, Var)]
            if vars_:
                return replace_random_subtree(tree, Const(random.gauss(0, 1)))
        return tree

    else:
        # Hoist: replace tree with a random subtree (shrink)
        all_nodes = tree.nodes()
        non_trivial = [n for n in all_nodes if isinstance(n, (UnaryOp, BinaryOp))]
        if non_trivial:
            return random.choice(non_trivial).copy()
        return tree.copy()


def crossover(tree1, tree2):
    """Subtree crossover: swap random subtrees between two parents."""
    child = tree1.copy()
    donor_nodes = tree2.nodes()
    if len(donor_nodes) > 0:
        donor_sub = random.choice(donor_nodes).copy()
        child = replace_random_subtree(child, donor_sub)
    return child


# ================================================================
# SAFE EVALUATION
# ================================================================
def safe_eval_curve(expr, z):
    """Evaluate expression on z values, handling numerical issues."""
    try:
        result = expr.eval(z)
        # Replace NaN/Inf
        result = torch.where(torch.isfinite(result), result, torch.zeros_like(result))
        result = torch.clamp(result, -50, 50)
        return result
    except Exception:
        return torch.zeros_like(z)


def curve_fingerprint(expr, n_points=200):
    """Compute normalized curve as a fingerprint for deduplication."""
    z = torch.linspace(-5, 5, n_points)
    y = safe_eval_curve(expr, z)

    # Check for degenerate (constant) output
    std = y.std().item()
    if std < 1e-6:
        return None, y.numpy()  # degenerate

    # Normalize for comparison
    y_norm = (y - y.mean()) / (std + 1e-8)
    return y_norm.numpy(), y.numpy()


def is_duplicate(fingerprint, existing_fingerprints, threshold=0.999):
    """Check if a curve is a duplicate of any existing one."""
    if fingerprint is None:
        return False  # degenerate curves are still cataloged
    fp = fingerprint / (np.linalg.norm(fingerprint) + 1e-8)
    for efp in existing_fingerprints:
        if efp is None:
            continue
        enorm = efp / (np.linalg.norm(efp) + 1e-8)
        sim = np.dot(fp, enorm)
        if abs(sim) > threshold:  # abs catches negated duplicates
            return True
    return False


# ================================================================
# MNIST CHARACTERIZATION
# ================================================================
def load_mnist(device):
    from torchvision import datasets, transforms
    print("Loading MNIST...")
    mnist = datasets.MNIST(root='./data', train=True, download=True,
                           transform=transforms.ToTensor())
    images = mnist.data.float() / 255.0
    images = images.view(-1, INPUT_DIM)
    labels = mnist.targets

    mu = images.mean(0, keepdim=True)
    std = images.std(0, keepdim=True).clamp(min=1e-6)
    images = (images - mu) / std

    torch.manual_seed(SEED)
    idx = torch.randperm(len(labels))
    images, labels = images[idx], labels[idx]

    n = int(len(labels) * TRAIN_FRAC)
    return images[:n].to(device), labels[:n].to(device)


def setup_characterization(device):
    """Create fixed random projections for fair activation comparison."""
    torch.manual_seed(SEED + 1000)
    weights = torch.randn(N_CHAR_NEURONS, INPUT_DIM, device=device) * (2.0 / INPUT_DIM**0.5)
    bias = torch.zeros(N_CHAR_NEURONS, device=device)
    return weights, bias


def characterize_activation(expr, train_x, train_labels, char_w, char_b, device):
    """Score an activation by MNIST k-NN accuracy with fixed random weights.

    Returns accuracy (float) and output stats dict.
    """
    n_samples = train_x.shape[0]
    accs = []

    for _ in range(N_EVAL_BATCHES):
        ix = torch.randint(0, n_samples, (EVAL_BATCH_SIZE,), device=device)
        batch = train_x[ix]  # [B, 784]
        labels = train_labels[ix]

        # Linear projection
        with torch.no_grad():
            linear = batch @ char_w.T + char_b  # [B, N_CHAR_NEURONS]

            # Apply activation (move to CPU for tree eval, back to GPU)
            # Actually, keep on same device as linear
            activated = safe_eval_curve(expr, linear)  # [B, N_CHAR_NEURONS]

            if not torch.isfinite(activated).all():
                activated = torch.where(torch.isfinite(activated), activated,
                                       torch.zeros_like(activated))

            # Check if output is degenerate
            if activated.std() < 1e-6:
                return 10.0, {'degenerate': True}

            # k-NN (split-half)
            half = EVAL_BATCH_SIZE // 2
            feat_norm = activated / (activated.norm(dim=-1, keepdim=True) + 1e-8)
            dists = torch.cdist(feat_norm[half:].unsqueeze(0),
                               feat_norm[:half].unsqueeze(0))[0]
            _, knn_idx = dists.topk(KNN_K, dim=-1, largest=False)
            knn_labels = labels[:half][knn_idx]
            class_ids = torch.arange(N_CLASSES, device=device)
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


# ================================================================
# KNOWN ACTIVATIONS (reference landmarks)
# ================================================================
def build_known_activations():
    """Build expression trees for known named activations."""
    x = Var()
    return {
        'identity': Var(),
        'abs': UnaryOp('abs', Var()),
        'neg': UnaryOp('neg', Var()),
        'square': BinaryOp('*', Var(), Var()),
        'exp': UnaryOp('exp', Var()),
        'sin': UnaryOp('sin', Var()),
        'cos': UnaryOp('cos', Var()),
        'gauss': UnaryOp('exp', UnaryOp('neg', BinaryOp('*', Var(), Var()))),
        'relu': BinaryOp('*', BinaryOp('+', Var(), UnaryOp('abs', Var())), Const(0.5)),
        'sigmoid': BinaryOp('/', Const(1),
                    BinaryOp('+', Const(1), UnaryOp('exp', UnaryOp('neg', Var())))),
        'softplus': UnaryOp('log', BinaryOp('+', Const(1), UnaryOp('exp', Var()))),
        'sin_sq': UnaryOp('sin', BinaryOp('*', Var(), Var())),
        'exp_neg_abs': UnaryOp('exp', UnaryOp('neg', UnaryOp('abs', Var()))),
        'x_sin_x': BinaryOp('*', Var(), UnaryOp('sin', Var())),
        'cos_sq': UnaryOp('cos', BinaryOp('*', Var(), Var())),
    }


# ================================================================
# EVOLUTION + CATALOG
# ================================================================
def evolve_and_catalog(train_x, train_labels, char_w, char_b, device):
    """GP evolution of activation functions. Catalogs every unique activation."""

    catalog = []       # list of {expr, expr_str, accuracy, curve, fingerprint, ...}
    fingerprints = []  # for dedup
    expr_strings = set()
    score_cache = {}   # expr_str -> accuracy

    def add_to_catalog(expr, accuracy, stats, gen=-1):
        expr_str = str(expr)
        fp, raw_curve = curve_fingerprint(expr)

        # Skip exact string duplicates
        if expr_str in expr_strings:
            return

        # Skip curve-shape duplicates
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
        })

    def evaluate(expr):
        """Evaluate and cache."""
        expr_str = str(expr)
        if expr_str in score_cache:
            return score_cache[expr_str], {'cached': True}
        acc, stats = characterize_activation(expr, train_x, train_labels,
                                             char_w, char_b, device)
        score_cache[expr_str] = acc
        return acc, stats

    # ---- Phase 0: Known activations ----
    print("\n=== Phase 0: Known activation landmarks ===")
    known = build_known_activations()
    for name, expr in known.items():
        acc, stats = evaluate(expr)
        add_to_catalog(expr, acc, stats, gen=-1)
        print(f"  {name:20s} = {str(expr):40s} -> {acc:.1f}%")

    # ---- Phase 1: GP Evolution ----
    print(f"\n=== Phase 1: GP Evolution ({POP_SIZE} pop × {GENS} gens) ===")

    # Initialize population
    population = []
    fitnesses = []
    for _ in range(POP_SIZE):
        depth = random.randint(1, MAX_DEPTH)
        tree = random_tree(depth)
        # Enforce size limit
        while tree.size() > MAX_NODES:
            tree = random_tree(depth)
        population.append(tree)
        acc, stats = evaluate(tree)
        fitnesses.append(acc)
        add_to_catalog(tree, acc, stats, gen=0)

    best_acc = max(fitnesses)
    best_idx = fitnesses.index(best_acc)
    print(f"  Initial best: {best_acc:.1f}% — {population[best_idx]}")
    print(f"  Catalog size: {len(catalog)}")

    t0 = time.time()
    for gen in range(1, GENS + 1):
        new_pop = []
        new_fit = []

        # Elitism: keep top 5%
        n_elite = max(1, int(POP_SIZE * 0.05))
        ranked = sorted(range(POP_SIZE), key=lambda i: fitnesses[i], reverse=True)
        for i in ranked[:n_elite]:
            new_pop.append(population[i].copy())
            new_fit.append(fitnesses[i])

        # Fill rest with offspring
        while len(new_pop) < POP_SIZE:
            if random.random() < MUTATION_RATE:
                # Tournament select parent, mutate
                candidates = random.sample(range(POP_SIZE), TOURNAMENT_SIZE)
                parent_idx = max(candidates, key=lambda i: fitnesses[i])
                child = mutate(population[parent_idx])
            else:
                # Tournament select two parents, crossover
                c1 = random.sample(range(POP_SIZE), TOURNAMENT_SIZE)
                c2 = random.sample(range(POP_SIZE), TOURNAMENT_SIZE)
                p1 = max(c1, key=lambda i: fitnesses[i])
                p2 = max(c2, key=lambda i: fitnesses[i])
                child = crossover(population[p1], population[p2])

            # Enforce size/depth limits
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
            best_idx = fitnesses.index(best_acc)

        if gen % 20 == 0 or gen == GENS:
            elapsed = time.time() - t0
            print(f"  gen {gen:>4}/{GENS} | best={best_acc:.1f}% | "
                  f"catalog={len(catalog)} unique | {elapsed:.0f}s")

    print(f"  Evolution done. Best: {best_acc:.1f}%")
    print(f"  Best expression: {population[fitnesses.index(best_acc)]}")

    # ---- Phase 2: Random exploration ----
    print(f"\n=== Phase 2: Random exploration ({N_RANDOM_EXTRA} trees) ===")
    for i in range(N_RANDOM_EXTRA):
        depth = random.randint(1, MAX_DEPTH)
        tree = random_tree(depth)
        while tree.size() > MAX_NODES:
            tree = random_tree(depth)
        acc, stats = evaluate(tree)
        add_to_catalog(tree, acc, stats, gen=-2)

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{N_RANDOM_EXTRA} explored, catalog={len(catalog)}")

    print(f"\n  TOTAL CATALOG: {len(catalog)} unique activations")
    n_degen = sum(1 for c in catalog if c['degenerate'])
    n_good = sum(1 for c in catalog if c['accuracy'] > 50)
    print(f"  Degenerate (constant output): {n_degen}")
    print(f"  Above 50% accuracy: {n_good}")
    print(f"  Best accuracy: {max(c['accuracy'] for c in catalog):.1f}%")

    return catalog


# ================================================================
# VISUALIZATION
# ================================================================
def visualize_catalog(catalog, out_dir):
    z = np.linspace(-5, 5, 200)

    # Sort by accuracy
    catalog_sorted = sorted(catalog, key=lambda c: c['accuracy'], reverse=True)

    # ---- Figure 1: Top 50 activations (detailed) ----
    top_n = min(50, len(catalog_sorted))
    top = [c for c in catalog_sorted if not c['degenerate']][:top_n]

    cols = 10
    rows = (top_n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(30, rows * 3.5))
    fig.suptitle(f"TOP {top_n} Evolved Activations (by MNIST accuracy)\n"
                 f"Total catalog: {len(catalog)} unique activations",
                 fontsize=18, fontweight='bold')

    for i in range(top_n):
        row, col = i // cols, i % cols
        ax = axes[row][col] if rows > 1 else axes[col]

        if i < len(top):
            entry = top[i]
            y = np.array(entry['curve'])
            acc = entry['accuracy']

            # Color by accuracy
            green = min(1.0, (acc - 10) / 80)
            color = (1 - green, green, 0.2)

            ax.plot(z, y, color=color, linewidth=2)
            ax.axhline(y=0, color='gray', linewidth=0.3, alpha=0.5)
            ax.axvline(x=0, color='gray', linewidth=0.3, alpha=0.5)

            # Truncate long expressions
            expr_str = entry['expression']
            if len(expr_str) > 35:
                expr_str = expr_str[:32] + '...'
            ax.set_title(f"#{i+1} {acc:.1f}%\n{expr_str}\nd={entry['depth']} n={entry['n_nodes']}",
                        fontsize=6, fontweight='bold')
            ax.set_xlim(-5, 5)
        else:
            ax.set_visible(False)
        ax.tick_params(labelsize=5)

    for i in range(top_n, rows * cols):
        row, col = i // cols, i % cols
        ax = axes[row][col] if rows > 1 else axes[col]
        ax.set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_dir / 'top_activations.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: top_activations.png")
    plt.close(fig)

    # ---- Figure 2: FULL catalog (tiny thumbnails) ----
    non_degen = [c for c in catalog_sorted if not c['degenerate']]
    n_show = min(500, len(non_degen))
    cols2 = 25
    rows2 = (n_show + cols2 - 1) // cols2
    fig, axes = plt.subplots(rows2, cols2, figsize=(50, rows2 * 2))
    fig.suptitle(f"PERIODIC TABLE: {n_show} Unique Non-Degenerate Activations\n"
                 f"(sorted by MNIST accuracy, green=high, red=low)",
                 fontsize=20, fontweight='bold')

    for i in range(n_show):
        row, col = i // cols2, i % cols2
        ax = axes[row][col] if rows2 > 1 else axes[col]

        entry = non_degen[i]
        y = np.array(entry['curve'])
        acc = entry['accuracy']
        green = min(1.0, max(0, (acc - 10) / 80))
        color = (1 - green, green, 0.2)

        ax.plot(z, y, color=color, linewidth=1)
        ax.set_xlim(-5, 5)
        ax.set_title(f"{acc:.0f}%", fontsize=5, color=color)
        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(n_show, rows2 * cols2):
        row, col = i // cols2, i % cols2
        ax = axes[row][col] if rows2 > 1 else axes[col]
        ax.set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / 'periodic_table.png', dpi=100, bbox_inches='tight')
    print(f"  Saved: periodic_table.png")
    plt.close(fig)

    # ---- Figure 3: Accuracy distribution ----
    accs = [c['accuracy'] for c in catalog if not c['degenerate']]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(accs, bins=50, color='#2196F3', edgecolor='white', alpha=0.8)
    ax.axvline(x=10, color='red', linestyle='--', linewidth=2, label='Random chance (10%)')

    # Mark known activations
    known_in_catalog = [c for c in catalog if c['generation'] == -1]
    for kc in known_in_catalog:
        if not kc['degenerate']:
            ax.axvline(x=kc['accuracy'], color='orange', linewidth=1, alpha=0.7)
            ax.text(kc['accuracy'], ax.get_ylim()[1] * 0.9,
                   kc['expression'][:20], fontsize=6, rotation=90,
                   va='top', ha='right', color='orange')

    ax.set_xlabel('MNIST Accuracy (%)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Accuracy Distribution of {len(accs)} Unique Activations\n'
                 f'(orange lines = known named activations)',
                 fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / 'accuracy_distribution.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: accuracy_distribution.png")
    plt.close(fig)

    # ---- Figure 4: Depth vs accuracy ----
    fig, ax = plt.subplots(figsize=(10, 6))
    depths = [c['depth'] for c in catalog if not c['degenerate']]
    accs_d = [c['accuracy'] for c in catalog if not c['degenerate']]
    ax.scatter(depths, accs_d, alpha=0.3, s=10, c=accs_d, cmap='RdYlGn')
    ax.set_xlabel('Expression Depth', fontsize=12)
    ax.set_ylabel('MNIST Accuracy (%)', fontsize=12)
    ax.set_title('Activation Complexity vs Performance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_dir / 'depth_vs_accuracy.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: depth_vs_accuracy.png")
    plt.close(fig)

    # ---- Figure 5: Best activations overlaid ----
    fig, ax = plt.subplots(figsize=(14, 8))
    top_overlay = [c for c in catalog_sorted if not c['degenerate']][:30]
    for i, entry in enumerate(top_overlay):
        y = np.array(entry['curve'])
        acc = entry['accuracy']
        green = min(1.0, max(0, (acc - 10) / 80))
        color = (1 - green, green, 0.2)
        label = f"{acc:.1f}% {entry['expression'][:25]}" if i < 10 else None
        ax.plot(z, y, color=color, linewidth=1.5, alpha=0.7, label=label)

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel('z', fontsize=12)
    ax.set_ylabel('f(z)', fontsize=12)
    ax.set_title('Top 30 Activations Overlaid', fontsize=14, fontweight='bold')
    ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(0, -0.08), ncol=2)
    ax.set_xlim(-5, 5)
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    fig.savefig(out_dir / 'top_overlaid.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: top_overlaid.png")
    plt.close(fig)


# ================================================================
# MAIN
# ================================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.set_float32_matmul_precision("high")

    sid = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("results") / f"primordial_{sid}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PRIMORDIAL ACTIVATION EVOLUTION")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Primordial ops:")
    print(f"    Unary:  {UNARY_OPS}")
    print(f"    Binary: {BINARY_OPS}")
    print(f"    Terminals: x, {CONSTANTS}")
    print(f"  Max depth: {MAX_DEPTH}, max nodes: {MAX_NODES}")
    print(f"  Pop: {POP_SIZE}, Gens: {GENS}, Random extra: {N_RANDOM_EXTRA}")
    print(f"  Characterization: {N_CHAR_NEURONS} neurons, k-NN k={KNN_K}")
    print(f"  Output: {out_dir}")

    t0 = time.time()

    train_x, train_labels = load_mnist(device)
    char_w, char_b = setup_characterization(device)
    print(f"  Train: {train_x.shape[0]} samples, {INPUT_DIM}D")

    catalog = evolve_and_catalog(train_x, train_labels, char_w, char_b, device)

    print(f"\nGenerating visualizations...")
    visualize_catalog(catalog, out_dir)

    # Save full catalog as JSON
    with open(out_dir / 'catalog.json', 'w') as f:
        json.dump({
            'n_activations': len(catalog),
            'primordial_ops': {
                'unary': UNARY_OPS,
                'binary': BINARY_OPS,
                'constants': CONSTANTS,
            },
            'config': {
                'pop_size': POP_SIZE,
                'gens': GENS,
                'max_depth': MAX_DEPTH,
                'max_nodes': MAX_NODES,
                'n_char_neurons': N_CHAR_NEURONS,
                'n_random_extra': N_RANDOM_EXTRA,
            },
            'activations': catalog,
        }, f, indent=2, default=str)
    print(f"  Saved: catalog.json ({len(catalog)} activations)")

    # Print top 20
    sorted_cat = sorted(catalog, key=lambda c: c['accuracy'], reverse=True)
    print(f"\n  TOP 20 ACTIVATIONS:")
    print(f"  {'#':>3} {'Acc':>7} {'Depth':>5} {'Nodes':>5}  Expression")
    print(f"  {'-'*80}")
    for i, entry in enumerate(sorted_cat[:20]):
        print(f"  {i+1:>3} {entry['accuracy']:>6.1f}% {entry['depth']:>5} "
              f"{entry['n_nodes']:>5}  {entry['expression']}")

    total_t = time.time() - t0
    print(f"\n  Total time: {total_t:.0f}s ({total_t/60:.1f} min)")
    print(f"  Output: {out_dir}")


if __name__ == "__main__":
    main()
