# Activation Galaxy Explorer

A tool for evolving and cataloging activation functions using genetic programming.

Instead of using fixed activation functions (ReLU, sigmoid, tanh), this evolves new ones from primordial operations (sin, cos, exp, log, +, -, *, /) and evaluates them across multiple datasets.

---

**What it does**

1. Downloads datasets (MNIST, FashionMNIST, EMNIST, CIFAR-10)
2. Seeds evolution with ~30 known activations (ReLU, sigmoid, swish, GELU, mish, etc.)
3. Evolves new activation functions via genetic programming, scored by k-NN classification accuracy
4. Collects all catalogs, deduplicates by curve similarity
5. Cross-scores every activation on every dataset
6. Renders a t-SNE visualization showing how activations cluster

---

**Output**

- `master_catalog.json` - All discovered activations with expressions, curves, and per-dataset scores
- `galaxy_*.png` - t-SNE visualization colored by source dataset or accuracy
- Per-dataset catalogs and basis expansion test results

---

**Usage**

```bash
# Full pipeline (takes a while)
python galaxy_explorer.py

# Quick test with smaller evolution
python galaxy_explorer.py --quick

# Skip evolution, just render from existing catalog
python galaxy_explorer.py --catalog path/to/catalog.json
```

---

**Activation format**

Each activation is stored as:
- Expression tree built from primordial ops
- 200-point curve (evaluated from -5 to 5)
- Classification scores on each dataset
- Properties (shape family, symmetry, saturation, monotonicity)

Example expressions:
```
(sin((x + 1)) - (x * 2))
(x / (1 + exp(neg(x))))
((x + abs(x)) * 0.5)
```

---

**What I've found so far**

Evolving activations across ~8 datasets (images, audio, text) produced a catalog of ~50K unique functions. Most are useless. Around 1-2% show up repeatedly as useful across tasks.

Some findings:
- Activations evolved for text classification transferred to image classification at 96% of native performance
- The `sin(f(x)) - 2x` family (oscillation minus linear baseline) kept appearing as a top performer across domains
- When visualized by curve similarity, activations cluster into distinct regions - specialists group by domain, generalists sit between them

The structure emerged from the data. I didn't impose it.

---

**Requirements**

```
torch
numpy
matplotlib
scikit-learn (for t-SNE)
torchvision (for dataset loading)
```

---

**Limitations**

- Evolution is slow. Full runs take hours.
- The catalog is biased toward classification tasks (that's what the fitness function rewards)
- Cross-dataset scores use simple k-NN, not the activations in actual trained networks
- Visualization is 2D t-SNE which loses information

---

**Files**

```
galaxy_explorer.py     # Main script - evolution + visualization
network_builder.py     # GUI for composing networks from catalog (optional)
evolve_primitives.py   # Standalone primordial evolution
basis_expansion.py     # Test catalog activations via greedy basis expansion
```

---

**Related work**

This is loosely inspired by:
- Neural Architecture Search (but for activations, not topology)
- Genetic Programming for symbolic regression
- Activation function search papers (Swish, GELU discovery)

The difference is cataloging everything explored, not just finding winners. The "failed" activations are still characterized - they tell you what doesn't work for which tasks.
