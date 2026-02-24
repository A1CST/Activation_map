#!/usr/bin/env python3
"""Network Builder - Drag-and-Drop Activation Network Composer.

Build neural networks by dragging activations from the master catalog into
layer slots, then train with evolutionary weight optimization.

Two modes per layer:
  Locked  - user picks exact activation, system evolves weights/bias only
  Palette - user picks allowed activations, evolution can swap between them

Usage: python network_builder.py [--catalog path/to/master_catalog.json]
"""

import os
import sys
import json
import time
import math
import threading
import queue
import argparse
from pathlib import Path
from collections import OrderedDict
from datetime import datetime

import numpy as np
import pygame

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

WINDOW_W, WINDOW_H = 1600, 900
FPS_TARGET = 60

BG_COLOR = (6, 8, 14)
PANEL_BG = (16, 20, 28)
PANEL_BORDER = (60, 80, 120)
CARD_BG = (24, 30, 42)
CARD_HOVER = (34, 42, 58)
CARD_SELECTED = (40, 60, 90)
LAYER_BG = (20, 28, 40)
LAYER_DROP = (40, 80, 60)
LAYER_BORDER = (80, 100, 140)
BTN_COLOR = (50, 65, 90)
BTN_HOVER = (65, 85, 120)
BTN_DISABLED = (30, 35, 45)
BTN_ACTIVE = (40, 120, 80)
DD_BG = (30, 38, 52)
DD_HOVER = (50, 60, 80)
TEXT_COLOR = (200, 210, 230)
TEXT_DIM = (120, 130, 150)
TEXT_BRIGHT = (230, 240, 255)
ACCENT = (80, 140, 220)
GREEN = (60, 180, 100)
RED = (200, 70, 70)
YELLOW = (200, 180, 60)
PROGRESS_BG = (30, 35, 45)
PROGRESS_FG = (60, 140, 200)

PALETTE_W = 380
STACK_W = 480
MONITOR_W = WINDOW_W - PALETTE_W - STACK_W
TOOLBAR_H = 44
STATUS_H = 28
CONTENT_Y = TOOLBAR_H
CONTENT_H = WINDOW_H - TOOLBAR_H - STATUS_H

MAX_LAYERS = 10
NEURON_CHOICES = [8, 16, 32, 64, 96, 128]
CARD_H = 52
LAYER_H = 100
DRAG_THRESHOLD = 5

TASK_KEYS = [
    "images_classification_CIFAR10",
    "images_classification_MNIST",
    "images_reconstruction_CIFAR10",
    "images_reconstruction_MNIST",
    "audio_classification",
    "audio_reconstruction",
    "text_classification",
    "text_reconstruction",
]

SORT_MODES = [
    ("Mean Score", None),
    ("CIFAR-10", 0),
    ("MNIST", 1),
    ("CIFAR-10 Recon", 2),
    ("MNIST Recon", 3),
    ("Audio Cls", 4),
    ("Audio Recon", 5),
    ("Text Cls", 6),
    ("Text Recon", 7),
]

DATASET_CONFIGS = {
    'MNIST': {
        'name': 'MNIST', 'loader': 'MNIST', 'input_dim': 784,
        'n_classes': 10, 'flatten': True, 'n_char': 32, 'n_basis': 32, 'max_elem': 10,
    },
    'FashionMNIST': {
        'name': 'FashionMNIST', 'loader': 'FashionMNIST', 'input_dim': 784,
        'n_classes': 10, 'flatten': True, 'n_char': 32, 'n_basis': 32, 'max_elem': 10,
    },
    'EMNIST_Digits': {
        'name': 'EMNIST_Digits', 'loader': 'EMNIST', 'input_dim': 784,
        'n_classes': 10, 'flatten': True, 'n_char': 32, 'n_basis': 32, 'max_elem': 10,
        'loader_kwargs': {'split': 'digits'}, 'max_samples': 60000,
    },
    'CIFAR10': {
        'name': 'CIFAR10', 'loader': 'CIFAR10', 'input_dim': 3072,
        'n_classes': 10, 'flatten': True, 'n_char': 96, 'n_basis': 64, 'max_elem': 15,
    },
    'SpeechCommands': {
        'custom': 'speech', 'input_dim': 4032,
        'n_classes': 35, 'n_char': 64, 'n_basis': 48, 'max_elem': 10,
    },
    'AG_News': {
        'custom': 'text', 'input_dim': 3000,
        'n_classes': 4, 'n_char': 48, 'n_basis': 48, 'max_elem': 10,
    },
}
DATASET_NAMES = list(DATASET_CONFIGS.keys())


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode('ascii', errors='replace').decode('ascii'), flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CatalogData
# ═══════════════════════════════════════════════════════════════════════════════

class CatalogData:
    """Loads master_catalog.json and pre-computes arrays for fast access."""

    def __init__(self, catalog_path, screen=None, font=None):
        raw = self._load_json(catalog_path, screen, font)
        activations = raw["activations"]

        # Filter non-degenerate
        good = [a for a in activations if not a.get("degenerate", False)]
        n = len(good)
        log(f"Catalog: {n} non-degenerate activations (of {len(activations)} total)")

        self.n = n
        self.expressions = [a["expression"] for a in good]
        self.expressions_lower = [e.lower() for e in self.expressions]
        self.evolved_for = [a.get("evolved_for", "unknown") for a in good]
        self.depths = np.array([a.get("depth", 0) for a in good], dtype=np.int32)
        self.n_nodes = np.array([a.get("n_nodes", 1) for a in good], dtype=np.int32)

        # Scores (N, 8)
        self.scores = np.zeros((n, 8), dtype=np.float32)
        for i, a in enumerate(good):
            sc = a.get("scores", {})
            for j, key in enumerate(TASK_KEYS):
                self.scores[i, j] = sc.get(key, 0.0)
        self.mean_scores = self.scores.mean(axis=1)

        # Curves (N, 200)
        self.curves = np.zeros((n, 200), dtype=np.float32)
        for i, a in enumerate(good):
            c = a.get("curve", [])
            L = min(len(c), 200)
            self.curves[i, :L] = c[:L]

        # Pre-sorted indices
        self.sorted_by_mean = np.argsort(-self.mean_scores)
        self.sorted_by_task = {}
        for j in range(8):
            self.sorted_by_task[j] = np.argsort(-self.scores[:, j])
        self.sorted_by_depth = np.argsort(self.depths)
        self.sorted_by_complexity = np.argsort(self.n_nodes)

    def _load_json(self, path, screen, font):
        if screen and font:
            screen.fill(BG_COLOR)
            txt = font.render("Loading catalog...", True, TEXT_COLOR)
            screen.blit(txt, (WINDOW_W // 2 - txt.get_width() // 2,
                              WINDOW_H // 2 - txt.get_height() // 2))
            pygame.display.flip()

        t0 = time.time()
        try:
            import orjson
            with open(path, 'rb') as f:
                data = orjson.loads(f.read())
            log(f"Loaded catalog with orjson in {time.time()-t0:.1f}s")
        except ImportError:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
            log(f"Loaded catalog with json in {time.time()-t0:.1f}s")
        return data


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CurveRenderer
# ═══════════════════════════════════════════════════════════════════════════════

class CurveRenderer:
    """Renders activation curve thumbnails with LRU cache."""

    def __init__(self, catalog):
        self.catalog = catalog
        self.cache = OrderedDict()
        self.max_cache = 2000

    def get(self, idx, width, height):
        key = (idx, width, height)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        surf = self._render(idx, width, height)
        self.cache[key] = surf
        if len(self.cache) > self.max_cache:
            self.cache.popitem(last=False)
        return surf

    def _render(self, idx, width, height):
        surf = pygame.Surface((width, height), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 0))
        curve = self.catalog.curves[idx]
        n = len(curve)
        if n < 2:
            return surf

        ymin, ymax = float(curve.min()), float(curve.max())
        if ymax - ymin < 1e-6:
            ymax = ymin + 1.0

        pad = 2
        w, h = width - 2 * pad, height - 2 * pad
        points = []
        step = max(1, n // width)
        for i in range(0, n, step):
            px = pad + (i / (n - 1)) * w
            py = pad + h - ((float(curve[i]) - ymin) / (ymax - ymin)) * h
            points.append((float(px), float(py)))

        if len(points) >= 2:
            pygame.draw.lines(surf, ACCENT, False, points, 2)

        # Zero line
        if ymin < 0 < ymax:
            zy = int(pad + h - ((0 - ymin) / (ymax - ymin)) * h)
            pygame.draw.line(surf, (60, 60, 80), (pad, zy), (pad + w, zy), 1)

        return surf


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Button
# ═══════════════════════════════════════════════════════════════════════════════

class Button:
    def __init__(self, rect, label, color=BTN_COLOR, hover_color=BTN_HOVER,
                 disabled_color=BTN_DISABLED):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.color = color
        self.hover_color = hover_color
        self.disabled_color = disabled_color
        self.enabled = True
        self.hovered = False

    def draw(self, screen, font):
        if not self.enabled:
            c = self.disabled_color
            tc = TEXT_DIM
        elif self.hovered:
            c = self.hover_color
            tc = TEXT_BRIGHT
        else:
            c = self.color
            tc = TEXT_COLOR
        pygame.draw.rect(screen, c, self.rect, border_radius=4)
        pygame.draw.rect(screen, PANEL_BORDER, self.rect, 1, border_radius=4)
        txt = font.render(self.label, True, tc)
        screen.blit(txt, (self.rect.centerx - txt.get_width() // 2,
                          self.rect.centery - txt.get_height() // 2))

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.enabled and self.rect.collidepoint(event.pos):
                return True
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Dropdown
# ═══════════════════════════════════════════════════════════════════════════════

class Dropdown:
    def __init__(self, rect, options, selected=0):
        self.rect = pygame.Rect(rect)
        self.options = options
        self.selected = selected
        self.is_open = False
        self.hovered_option = -1
        self.item_h = self.rect.height

    @property
    def value(self):
        return self.options[self.selected]

    def draw(self, screen, font):
        # Closed state
        c = DD_HOVER if self.is_open else DD_BG
        pygame.draw.rect(screen, c, self.rect, border_radius=3)
        pygame.draw.rect(screen, PANEL_BORDER, self.rect, 1, border_radius=3)
        txt = font.render(self.options[self.selected], True, TEXT_COLOR)
        screen.blit(txt, (self.rect.x + 6, self.rect.centery - txt.get_height() // 2))
        # Arrow
        ax = self.rect.right - 14
        ay = self.rect.centery
        pygame.draw.polygon(screen, TEXT_DIM, [(ax - 4, ay - 3), (ax + 4, ay - 3), (ax, ay + 3)])

    def draw_overlay(self, screen, font):
        """Draw expanded list (call after everything else for z-order)."""
        if not self.is_open:
            return
        n = len(self.options)
        list_rect = pygame.Rect(self.rect.x, self.rect.bottom,
                                self.rect.width, n * self.item_h)
        pygame.draw.rect(screen, DD_BG, list_rect)
        pygame.draw.rect(screen, PANEL_BORDER, list_rect, 1)
        for i, opt in enumerate(self.options):
            oy = self.rect.bottom + i * self.item_h
            item_rect = pygame.Rect(self.rect.x, oy, self.rect.width, self.item_h)
            if i == self.hovered_option:
                pygame.draw.rect(screen, DD_HOVER, item_rect)
            txt = font.render(opt, True, TEXT_BRIGHT if i == self.selected else TEXT_COLOR)
            screen.blit(txt, (item_rect.x + 6, item_rect.centery - txt.get_height() // 2))

    def handle_event(self, event):
        """Returns True if selection changed."""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.is_open:
                # Check if clicked on an option
                n = len(self.options)
                list_rect = pygame.Rect(self.rect.x, self.rect.bottom,
                                        self.rect.width, n * self.item_h)
                if list_rect.collidepoint(event.pos):
                    idx = (event.pos[1] - self.rect.bottom) // self.item_h
                    if 0 <= idx < n:
                        old = self.selected
                        self.selected = idx
                        self.is_open = False
                        return old != idx
                self.is_open = False
                return False
            elif self.rect.collidepoint(event.pos):
                self.is_open = True
                return False
        elif event.type == pygame.MOUSEMOTION and self.is_open:
            n = len(self.options)
            list_rect = pygame.Rect(self.rect.x, self.rect.bottom,
                                    self.rect.width, n * self.item_h)
            if list_rect.collidepoint(event.pos):
                self.hovered_option = (event.pos[1] - self.rect.bottom) // self.item_h
            else:
                self.hovered_option = -1
        return False

    def close(self):
        self.is_open = False
        self.hovered_option = -1


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ActivationPalette
# ═══════════════════════════════════════════════════════════════════════════════

class ActivationPalette:
    """Left panel: scrollable list of activation cards."""

    def __init__(self, catalog, curve_renderer):
        self.catalog = catalog
        self.renderer = curve_renderer
        self.rect = pygame.Rect(0, CONTENT_Y, PALETTE_W, CONTENT_H)

        # Search
        self.search_text = ""
        self.search_active = False
        self.search_rect = pygame.Rect(8, CONTENT_Y + 6, PALETTE_W - 80, 24)

        # Sort
        self.sort_dd = Dropdown(
            (PALETTE_W - 68, CONTENT_Y + 6, 62, 24),
            [s[0] for s in SORT_MODES], 0)

        # Scroll
        self.scroll_y = 0
        self.filtered_indices = list(range(catalog.n))
        self._apply_sort()
        self.card_area_y = CONTENT_Y + 36
        self.card_area_h = CONTENT_H - 36

        # Drag state
        self.drag_start_pos = None
        self.drag_start_idx = -1

    def _apply_sort(self):
        mode_idx = self.sort_dd.selected
        _, task_key = SORT_MODES[mode_idx]

        if task_key is None:
            order = np.argsort(-self.catalog.mean_scores)
        else:
            order = np.argsort(-self.catalog.scores[:, task_key])

        if self.search_text:
            query = self.search_text.lower()
            allowed = set()
            for i in range(self.catalog.n):
                if query in self.catalog.expressions_lower[i]:
                    allowed.add(i)
            self.filtered_indices = [int(i) for i in order if i in allowed]
        else:
            self.filtered_indices = [int(i) for i in order]
        self.scroll_y = 0

    def handle_event(self, event):
        """Returns (drag_idx, pos) if drag started, else None."""
        # Sort dropdown
        if self.sort_dd.handle_event(event):
            self._apply_sort()
            return None

        if self.sort_dd.is_open:
            return None

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.search_rect.collidepoint(event.pos):
                self.search_active = True
                return None
            else:
                self.search_active = False

            # Check card click for drag start
            if self.rect.collidepoint(event.pos) and event.pos[1] >= self.card_area_y:
                card_idx = self._card_at(event.pos)
                if card_idx >= 0:
                    self.drag_start_pos = event.pos
                    self.drag_start_idx = card_idx
                return None

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.drag_start_pos = None
            self.drag_start_idx = -1

        elif event.type == pygame.MOUSEMOTION:
            if self.drag_start_pos and self.drag_start_idx >= 0:
                dx = abs(event.pos[0] - self.drag_start_pos[0])
                dy = abs(event.pos[1] - self.drag_start_pos[1])
                if dx + dy >= DRAG_THRESHOLD:
                    idx = self.drag_start_idx
                    self.drag_start_pos = None
                    self.drag_start_idx = -1
                    return (idx, event.pos)

        elif event.type == pygame.MOUSEWHEEL:
            if self.rect.collidepoint(pygame.mouse.get_pos()):
                self.scroll_y -= event.y * 40
                max_scroll = max(0, len(self.filtered_indices) * CARD_H - self.card_area_h)
                self.scroll_y = max(0, min(self.scroll_y, max_scroll))

        elif event.type == pygame.KEYDOWN and self.search_active:
            if event.key == pygame.K_BACKSPACE:
                self.search_text = self.search_text[:-1]
                self._apply_sort()
            elif event.key == pygame.K_ESCAPE:
                self.search_active = False
                self.search_text = ""
                self._apply_sort()
            elif event.key == pygame.K_RETURN:
                self.search_active = False
            elif event.unicode and event.unicode.isprintable():
                self.search_text += event.unicode
                self._apply_sort()

        return None

    def _card_at(self, pos):
        """Return catalog index of card at pos, or -1."""
        ry = pos[1] - self.card_area_y + self.scroll_y
        if ry < 0:
            return -1
        row = int(ry // CARD_H)
        if 0 <= row < len(self.filtered_indices):
            return self.filtered_indices[row]
        return -1

    def draw(self, screen, font, small_font):
        # Panel background
        pygame.draw.rect(screen, PANEL_BG, self.rect)
        pygame.draw.line(screen, PANEL_BORDER,
                         (self.rect.right - 1, self.rect.top),
                         (self.rect.right - 1, self.rect.bottom))

        # Search box
        sc = ACCENT if self.search_active else PANEL_BORDER
        pygame.draw.rect(screen, (12, 16, 24), self.search_rect)
        pygame.draw.rect(screen, sc, self.search_rect, 1)
        if self.search_text:
            st = font.render(self.search_text, True, TEXT_COLOR)
        else:
            st = font.render("Search...", True, TEXT_DIM)
        screen.blit(st, (self.search_rect.x + 4, self.search_rect.y + 3))

        # Sort dropdown (closed state)
        self.sort_dd.draw(screen, small_font)

        # Cards (clipped)
        clip = pygame.Rect(0, self.card_area_y, PALETTE_W, self.card_area_h)
        screen.set_clip(clip)

        first_visible = max(0, self.scroll_y // CARD_H)
        last_visible = min(len(self.filtered_indices),
                           first_visible + self.card_area_h // CARD_H + 2)
        mouse_pos = pygame.mouse.get_pos()

        for row in range(first_visible, last_visible):
            idx = self.filtered_indices[row]
            cy = self.card_area_y + row * CARD_H - self.scroll_y
            card_rect = pygame.Rect(4, cy, PALETTE_W - 12, CARD_H - 2)

            hovered = card_rect.collidepoint(mouse_pos)
            bg = CARD_HOVER if hovered else CARD_BG
            pygame.draw.rect(screen, bg, card_rect, border_radius=3)

            # Curve thumbnail
            thumb = self.renderer.get(idx, 56, 34)
            screen.blit(thumb, (card_rect.x + 4, card_rect.y + 8))

            # Expression (truncated)
            expr = self.catalog.expressions[idx]
            if len(expr) > 30:
                expr = expr[:27] + "..."
            et = font.render(expr, True, TEXT_COLOR)
            screen.blit(et, (card_rect.x + 64, card_rect.y + 4))

            # Mean score + origin
            score_str = f"mean={self.catalog.mean_scores[idx]:.1f}"
            origin = self.catalog.evolved_for[idx]
            if len(origin) > 12:
                origin = origin[:12]
            info = small_font.render(f"{score_str}  {origin}", True, TEXT_DIM)
            screen.blit(info, (card_rect.x + 64, card_rect.y + 22))

            # Depth indicator
            d = self.catalog.depths[idx]
            dt = small_font.render(f"d={d}", True, TEXT_DIM)
            screen.blit(dt, (card_rect.x + 64, card_rect.y + 36))

        screen.set_clip(None)

        # Count
        ct = small_font.render(f"{len(self.filtered_indices)} activations", True, TEXT_DIM)
        screen.blit(ct, (8, self.rect.bottom - 18))


# ═══════════════════════════════════════════════════════════════════════════════
# 6. LayerSlot
# ═══════════════════════════════════════════════════════════════════════════════

class LayerSlot:
    """One layer in the network stack."""

    def __init__(self, layer_num, activation_idx=-1, n_neurons=32, mode="locked"):
        self.layer_num = layer_num
        self.activation_idx = activation_idx  # primary activation (catalog idx)
        self.palette_indices = []  # for palette mode
        self.mode = mode  # "locked" or "palette"
        self.n_neurons = n_neurons
        self.result_baseline = None
        self.result_fitness = None
        self.result_expr = None


# ═══════════════════════════════════════════════════════════════════════════════
# 7. NetworkStack
# ═══════════════════════════════════════════════════════════════════════════════

class NetworkStack:
    """Center panel: vertical list of LayerSlots."""

    def __init__(self, catalog, curve_renderer):
        self.catalog = catalog
        self.renderer = curve_renderer
        self.layers = []
        self.rect = pygame.Rect(PALETTE_W, CONTENT_Y, STACK_W, CONTENT_H)
        self.scroll_y = 0
        self.locked = False  # locked during training
        self.hovered_layer = -1
        self.drop_target = -1  # layer index being hovered during drag, or len(layers) for "add"

        # Neuron dropdowns (created per layer)
        self.neuron_dds = []

        # Add-layer button region
        self.add_btn_rect = pygame.Rect(0, 0, 0, 0)  # updated in draw

    def add_layer(self, activation_idx, n_neurons=32, mode="locked"):
        if len(self.layers) >= MAX_LAYERS:
            return False
        slot = LayerSlot(len(self.layers) + 1, activation_idx, n_neurons, mode)
        self.layers.append(slot)
        # Add neuron dropdown
        dd = Dropdown((0, 0, 60, 22), [str(n) for n in NEURON_CHOICES],
                      NEURON_CHOICES.index(n_neurons) if n_neurons in NEURON_CHOICES else 2)
        self.neuron_dds.append(dd)
        return True

    def remove_layer(self, idx):
        if 0 <= idx < len(self.layers):
            self.layers.pop(idx)
            self.neuron_dds.pop(idx)
            for i, L in enumerate(self.layers):
                L.layer_num = i + 1

    def get_total_dim(self):
        return sum(L.n_neurons for L in self.layers)

    def get_config(self):
        """Return config list for training thread."""
        cfg = []
        for L in self.layers:
            d = {
                'activation_idx': L.activation_idx,
                'palette_indices': list(L.palette_indices),
                'mode': L.mode,
                'n_neurons': L.n_neurons,
            }
            cfg.append(d)
        return cfg

    def handle_event(self, event, dragging=False):
        """Returns action string or None. Sets drop_target during drag."""
        if self.locked:
            return None

        # Handle neuron dropdowns
        for i, dd in enumerate(self.neuron_dds):
            if dd.handle_event(event):
                self.layers[i].n_neurons = NEURON_CHOICES[dd.selected]
                return None
            if dd.is_open:
                return None

        if event.type == pygame.MOUSEWHEEL and self.rect.collidepoint(pygame.mouse.get_pos()):
            self.scroll_y -= event.y * 40
            max_scroll = max(0, len(self.layers) * LAYER_H + 40 - self.rect.height)
            self.scroll_y = max(0, min(self.scroll_y, max_scroll))

        if event.type == pygame.MOUSEMOTION:
            if dragging and self.rect.collidepoint(event.pos):
                self.drop_target = self._layer_at(event.pos)
            elif dragging:
                self.drop_target = -1
            self.hovered_layer = self._layer_at(event.pos) if self.rect.collidepoint(event.pos) else -1

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if not self.rect.collidepoint(event.pos):
                return None
            ly = self._layer_at(event.pos)

            # Check remove (X) button
            if 0 <= ly < len(self.layers):
                lx = self.rect.x + STACK_W - 30
                lay_y = self.rect.y + ly * LAYER_H - self.scroll_y + 4
                x_btn = pygame.Rect(lx, lay_y, 22, 22)
                if x_btn.collidepoint(event.pos):
                    self.remove_layer(ly)
                    return "removed"

                # Check mode toggle
                mode_rect = pygame.Rect(self.rect.x + 10, lay_y + 72, 70, 18)
                if mode_rect.collidepoint(event.pos):
                    L = self.layers[ly]
                    L.mode = "palette" if L.mode == "locked" else "locked"
                    if L.mode == "palette" and L.activation_idx >= 0:
                        if L.activation_idx not in L.palette_indices:
                            L.palette_indices.append(L.activation_idx)
                    return "mode_changed"

            # Add layer area
            if ly == len(self.layers):
                if len(self.layers) < MAX_LAYERS:
                    return "add_empty"

        # Right-click to remove palette item
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
            if self.rect.collidepoint(event.pos):
                ly = self._layer_at(event.pos)
                if 0 <= ly < len(self.layers) and self.layers[ly].mode == "palette":
                    # Check if clicking on a palette chip
                    L = self.layers[ly]
                    lay_y = self.rect.y + ly * LAYER_H - self.scroll_y + 4
                    chip_x = self.rect.x + 100
                    chip_y = lay_y + 50
                    for pi, pidx in enumerate(L.palette_indices):
                        cr = pygame.Rect(chip_x + pi * 56, chip_y, 52, 18)
                        if cr.collidepoint(event.pos):
                            L.palette_indices.pop(pi)
                            return "palette_removed"

        return None

    def _layer_at(self, pos):
        """Return layer index at pos, or len(layers) for add-area, or -1."""
        if not self.rect.collidepoint(pos):
            return -1
        ry = pos[1] - self.rect.y + self.scroll_y
        idx = int(ry // LAYER_H)
        if 0 <= idx < len(self.layers):
            return idx
        if idx == len(self.layers) and len(self.layers) < MAX_LAYERS:
            return len(self.layers)
        return -1

    def handle_drop(self, catalog_idx):
        """Handle dropping an activation. Returns True if accepted."""
        t = self.drop_target
        self.drop_target = -1
        if t < 0:
            return False

        if t < len(self.layers):
            L = self.layers[t]
            if L.mode == "locked":
                L.activation_idx = catalog_idx
            else:
                if catalog_idx not in L.palette_indices:
                    L.palette_indices.append(catalog_idx)
                if L.activation_idx < 0:
                    L.activation_idx = catalog_idx
            return True
        elif t == len(self.layers):
            return self.add_layer(catalog_idx)
        return False

    def draw(self, screen, font, small_font, dragging=False):
        # Panel background
        pygame.draw.rect(screen, PANEL_BG, self.rect)
        pygame.draw.line(screen, PANEL_BORDER,
                         (self.rect.right - 1, self.rect.top),
                         (self.rect.right - 1, self.rect.bottom))

        clip = self.rect.copy()
        screen.set_clip(clip)

        for i, L in enumerate(self.layers):
            ly = self.rect.y + i * LAYER_H - self.scroll_y
            if ly + LAYER_H < self.rect.y or ly > self.rect.bottom:
                continue

            layer_rect = pygame.Rect(self.rect.x + 6, ly + 2, STACK_W - 16, LAYER_H - 4)

            # Drop highlight
            if dragging and self.drop_target == i:
                pygame.draw.rect(screen, LAYER_DROP, layer_rect, border_radius=5)
            else:
                pygame.draw.rect(screen, LAYER_BG, layer_rect, border_radius=5)
            pygame.draw.rect(screen, LAYER_BORDER, layer_rect, 1, border_radius=5)

            # Layer label
            lt = font.render(f"Layer {L.layer_num}", True, ACCENT)
            screen.blit(lt, (layer_rect.x + 8, layer_rect.y + 6))

            # Curve + expression
            if L.activation_idx >= 0:
                thumb = self.renderer.get(L.activation_idx, 72, 42)
                screen.blit(thumb, (layer_rect.x + 8, layer_rect.y + 26))
                expr = self.catalog.expressions[L.activation_idx]
                if len(expr) > 28:
                    expr = expr[:25] + "..."
                et = font.render(expr, True, TEXT_COLOR)
                screen.blit(et, (layer_rect.x + 84, layer_rect.y + 28))
            else:
                et = font.render("(empty - drag here)", True, TEXT_DIM)
                screen.blit(et, (layer_rect.x + 84, layer_rect.y + 28))

            # Neuron dropdown
            dd = self.neuron_dds[i]
            dd.rect.x = layer_rect.x + 84
            dd.rect.y = layer_rect.y + 50
            dd.draw(screen, small_font)

            nt = small_font.render("neurons:", True, TEXT_DIM)
            screen.blit(nt, (layer_rect.x + 84 + 64, layer_rect.y + 54))

            # Mode indicator
            mode_rect = pygame.Rect(layer_rect.x + 10, layer_rect.y + 72, 70, 18)
            mc = GREEN if L.mode == "locked" else YELLOW
            pygame.draw.rect(screen, mc, mode_rect, 1, border_radius=3)
            mt = small_font.render(L.mode.capitalize(), True, mc)
            screen.blit(mt, (mode_rect.x + 4, mode_rect.y + 2))

            # Palette chips
            if L.mode == "palette" and L.palette_indices:
                chip_x = layer_rect.x + 100
                chip_y = layer_rect.y + 72
                pc = small_font.render(f"({len(L.palette_indices)})", True, TEXT_DIM)
                screen.blit(pc, (chip_x - 18, chip_y + 2))
                for pi, pidx in enumerate(L.palette_indices[:5]):
                    cr = pygame.Rect(chip_x + pi * 56, chip_y, 52, 18)
                    pygame.draw.rect(screen, CARD_BG, cr, border_radius=2)
                    expr_short = self.catalog.expressions[pidx][:6]
                    ct = small_font.render(expr_short, True, TEXT_COLOR)
                    screen.blit(ct, (cr.x + 2, cr.y + 2))

            # Result (if trained)
            if L.result_fitness is not None:
                rt = small_font.render(
                    f"{L.result_baseline:.1f}% -> {L.result_fitness:.1f}%",
                    True, GREEN)
                screen.blit(rt, (layer_rect.right - 120, layer_rect.y + 8))

            # Remove (X)
            if not self.locked:
                x_rect = pygame.Rect(layer_rect.right - 24, layer_rect.y + 4, 22, 22)
                pygame.draw.rect(screen, RED, x_rect, 1, border_radius=3)
                xt = font.render("x", True, RED)
                screen.blit(xt, (x_rect.x + 6, x_rect.y + 2))

        # Add layer area
        n = len(self.layers)
        if n < MAX_LAYERS:
            aly = self.rect.y + n * LAYER_H - self.scroll_y + 4
            self.add_btn_rect = pygame.Rect(self.rect.x + 6, aly, STACK_W - 16, 36)
            if dragging and self.drop_target == n:
                pygame.draw.rect(screen, LAYER_DROP, self.add_btn_rect, border_radius=5)
            else:
                pygame.draw.rect(screen, CARD_BG, self.add_btn_rect, border_radius=5)
            pygame.draw.rect(screen, PANEL_BORDER, self.add_btn_rect, 1, border_radius=5)
            at = font.render("+ Add Layer", True, TEXT_DIM)
            screen.blit(at, (self.add_btn_rect.centerx - at.get_width() // 2,
                             self.add_btn_rect.centery - at.get_height() // 2))

        screen.set_clip(None)

        # Info bar at bottom
        info = f"Layers: {n}/{MAX_LAYERS}   Total dim: {self.get_total_dim()}"
        it = small_font.render(info, True, TEXT_DIM)
        screen.blit(it, (self.rect.x + 8, self.rect.bottom - 18))

    def draw_overlays(self, screen, font):
        """Draw open dropdowns on top."""
        for dd in self.neuron_dds:
            dd.draw_overlay(screen, font)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. DragController
# ═══════════════════════════════════════════════════════════════════════════════

class DragController:
    """Manages drag-and-drop from palette to stack."""

    def __init__(self, curve_renderer):
        self.renderer = curve_renderer
        self.dragging = False
        self.drag_idx = -1  # catalog index
        self.pos = (0, 0)

    def start(self, catalog_idx, pos):
        self.dragging = True
        self.drag_idx = catalog_idx
        self.pos = pos

    def update(self, pos):
        if self.dragging:
            self.pos = pos

    def stop(self):
        idx = self.drag_idx
        self.dragging = False
        self.drag_idx = -1
        return idx

    def draw(self, screen, font):
        if not self.dragging or self.drag_idx < 0:
            return
        # Ghost thumbnail
        thumb = self.renderer.get(self.drag_idx, 72, 42)
        ghost = thumb.copy()
        ghost.set_alpha(180)
        screen.blit(ghost, (self.pos[0] - 36, self.pos[1] - 21))


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Toolbar
# ═══════════════════════════════════════════════════════════════════════════════

class Toolbar:
    """Top bar with dataset selector and control buttons."""

    def __init__(self):
        self.rect = pygame.Rect(0, 0, WINDOW_W, TOOLBAR_H)
        self.dataset_dd = Dropdown((8, 10, 140, 24), DATASET_NAMES, 0)
        self.train_btn = Button((340, 8, 60, 28), "Train", color=BTN_ACTIVE)
        self.stop_btn = Button((406, 8, 50, 28), "Stop", color=RED)
        self.stop_btn.enabled = False
        self.reset_btn = Button((462, 8, 54, 28), "Reset")

    @property
    def dataset_name(self):
        return DATASET_NAMES[self.dataset_dd.selected]

    @property
    def dataset_config(self):
        return DATASET_CONFIGS[self.dataset_name]

    def set_training(self, active):
        self.train_btn.enabled = not active
        self.stop_btn.enabled = active
        self.dataset_dd.close()

    def handle_event(self, event):
        """Returns action string or None."""
        if self.dataset_dd.handle_event(event):
            return "dataset_changed"
        if self.dataset_dd.is_open:
            return None
        if self.train_btn.handle_event(event):
            return "train"
        if self.stop_btn.handle_event(event):
            return "stop"
        if self.reset_btn.handle_event(event):
            return "reset"
        return None

    def draw(self, screen, font, small_font):
        pygame.draw.rect(screen, PANEL_BG, self.rect)
        pygame.draw.line(screen, PANEL_BORDER, (0, TOOLBAR_H - 1), (WINDOW_W, TOOLBAR_H - 1))

        # Dataset dropdown
        dl = font.render("Dataset:", True, TEXT_DIM)
        screen.blit(dl, (160, 14))
        self.dataset_dd.rect.x = 220
        self.dataset_dd.draw(screen, font)

        # Buttons
        self.train_btn.draw(screen, font)
        self.stop_btn.draw(screen, font)
        self.reset_btn.draw(screen, font)

        # Dataset info
        cfg = self.dataset_config
        dim = cfg['input_dim']
        cls = cfg['n_classes']
        info = f"{dim}D, {cls} classes"
        it = small_font.render(info, True, TEXT_DIM)
        screen.blit(it, (530, 16))

    def draw_overlays(self, screen, font):
        self.dataset_dd.draw_overlay(screen, font)


# ═══════════════════════════════════════════════════════════════════════════════
# 10. TrainingMonitor
# ═══════════════════════════════════════════════════════════════════════════════

class TrainingMonitor:
    """Right panel showing live training progress."""

    def __init__(self):
        self.rect = pygame.Rect(PALETTE_W + STACK_W, CONTENT_Y, MONITOR_W, CONTENT_H)
        self.status = "Ready"
        self.current_layer = 0
        self.gen = 0
        self.gen_limit = 0
        self.stag = 0
        self.stag_limit = 0
        self.fitness = 0.0
        self.best_fitness = 0.0
        self.baseline = 0.0
        self.layer_history = []  # list of (layer_num, baseline, final, expr)
        self.train_acc = None
        self.test_acc = None
        self.log_lines = []
        self.scroll_y = 0

    def reset(self):
        self.status = "Ready"
        self.current_layer = 0
        self.gen = 0
        self.gen_limit = 0
        self.stag = 0
        self.stag_limit = 0
        self.fitness = 0.0
        self.best_fitness = 0.0
        self.baseline = 0.0
        self.layer_history = []
        self.train_acc = None
        self.test_acc = None
        self.log_lines = []
        self.scroll_y = 0

    def add_log(self, msg):
        self.log_lines.append(msg)
        if len(self.log_lines) > 200:
            self.log_lines = self.log_lines[-100:]

    def draw(self, screen, font, small_font):
        # Panel background
        pygame.draw.rect(screen, PANEL_BG, self.rect)

        x0 = self.rect.x + 12
        y = self.rect.y + 8

        # Status
        st = font.render(f"Status: {self.status}", True, ACCENT)
        screen.blit(st, (x0, y)); y += 22

        if self.current_layer > 0:
            # Generation info
            gt = font.render(f"Gen: {self.gen} / {self.gen_limit}", True, TEXT_COLOR)
            screen.blit(gt, (x0, y)); y += 18
            stt = font.render(f"Stag: {self.stag} / {self.stag_limit}", True, TEXT_COLOR)
            screen.blit(stt, (x0, y)); y += 22

            # Progress bar
            bar_w = MONITOR_W - 24
            bar_rect = pygame.Rect(x0, y, bar_w, 14)
            pygame.draw.rect(screen, PROGRESS_BG, bar_rect, border_radius=3)
            if self.stag_limit > 0:
                frac = min(1.0, self.stag / self.stag_limit)
                fill_w = int(frac * bar_w)
                fill_rect = pygame.Rect(x0, y, fill_w, 14)
                pygame.draw.rect(screen, PROGRESS_FG, fill_rect, border_radius=3)
                pt = small_font.render(f"{int(frac*100)}%", True, TEXT_BRIGHT)
                screen.blit(pt, (x0 + bar_w // 2 - pt.get_width() // 2, y + 1))
            y += 22

            # Fitness
            ft = font.render(f"Fitness: {self.fitness:.1f}%", True, GREEN)
            screen.blit(ft, (x0, y)); y += 18
            bt = font.render(f"Best:    {self.best_fitness:.1f}%", True, TEXT_COLOR)
            screen.blit(bt, (x0, y)); y += 18
            bl = font.render(f"Baseline: {self.baseline:.1f}%", True, TEXT_DIM)
            screen.blit(bl, (x0, y)); y += 26

        # Layer history
        if self.layer_history:
            ht = font.render("Layer History:", True, ACCENT)
            screen.blit(ht, (x0, y)); y += 18
            for lnum, base, final, expr in self.layer_history:
                if len(expr) > 22:
                    expr = expr[:19] + "..."
                color = GREEN if final > base else RED
                lt = small_font.render(
                    f"L{lnum}: {base:.1f}% -> {final:.1f}% [{expr}]", True, color)
                screen.blit(lt, (x0 + 4, y)); y += 16
            y += 8

        # Eval results
        if self.train_acc is not None:
            et = font.render("Final Eval (held-out):", True, ACCENT)
            screen.blit(et, (x0, y)); y += 20
            ta = font.render(f"  Train: {self.train_acc:.1f}%", True, TEXT_COLOR)
            screen.blit(ta, (x0, y)); y += 18
            if self.test_acc is not None:
                te = font.render(f"  Test:  {self.test_acc:.1f}%", True, GREEN)
                screen.blit(te, (x0, y)); y += 18
            y += 8

        # Log lines (bottom section)
        log_y = max(y + 8, self.rect.y + self.rect.height - 200)
        lt = small_font.render("Log:", True, TEXT_DIM)
        screen.blit(lt, (x0, log_y)); log_y += 14

        clip = pygame.Rect(x0, log_y, MONITOR_W - 24, self.rect.bottom - log_y - 4)
        screen.set_clip(clip)

        # Show last N lines that fit
        max_lines = clip.height // 13
        start = max(0, len(self.log_lines) - max_lines)
        for i in range(start, len(self.log_lines)):
            line = self.log_lines[i]
            if len(line) > 55:
                line = line[:52] + "..."
            lt = small_font.render(line, True, TEXT_DIM)
            screen.blit(lt, (x0, log_y))
            log_y += 13

        screen.set_clip(None)


# ═══════════════════════════════════════════════════════════════════════════════
# 11. TrainingThread
# ═══════════════════════════════════════════════════════════════════════════════

class TrainingThread(threading.Thread):
    """Background thread that runs evolutionary basis expansion."""

    def __init__(self, layer_configs, dataset_name, catalog_curves, catalog_exprs,
                 msg_queue):
        super().__init__(daemon=True)
        self.layer_configs = layer_configs
        self.dataset_name = dataset_name
        self.catalog_curves = catalog_curves  # numpy (N, 200)
        self.catalog_exprs = catalog_exprs    # list[str]
        self.msg_queue = msg_queue
        self.stop_event = threading.Event()

    def post(self, msg):
        self.msg_queue.put(msg)

    def run(self):
        try:
            self._run_impl()
        except Exception as e:
            self.post({"type": "error", "msg": str(e)})

    def _run_impl(self):
        import torch
        # Lazy import from sibling modules
        sys.path.insert(0, str(Path(__file__).parent))
        from dataset_experiment import (
            load_dataset, setup_characterization, knn_fitness,
            batched_knn_eval, apply_activation_batch, apply_activation_single,
            basis_forward, screen_activations,
            SEED, KNN_K, BASIS_POP_SIZE, BASIS_BATCH_SIZE, BASIS_ELITE_FRAC,
            BASIS_MUT_RATE, BASIS_MUT_SCALE, BASIS_STAG_FIRST, BASIS_STAG_REST,
            BASIS_SCREEN_TRIALS, BASIS_SCREEN_TOP_K,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.post({"type": "log", "msg": f"Device: {device}"})

        # ── Load dataset ──
        self.post({"type": "status", "msg": "Loading dataset..."})
        ds_cfg = DATASET_CONFIGS[self.dataset_name].copy()

        if 'custom' in ds_cfg:
            custom = ds_cfg['custom']
            if custom == 'speech':
                from master_catalog import load_audio_speechcommands
                features, labels, n_classes = load_audio_speechcommands(device)
            elif custom == 'text':
                from master_catalog import load_text_agnews
                features, labels, n_classes = load_text_agnews(device)
            else:
                self.post({"type": "error", "msg": f"Unknown custom loader: {custom}"})
                return

            input_dim = features.shape[1]
            # Split train/eval
            torch.manual_seed(SEED)
            idx = torch.randperm(len(labels))
            features, labels = features[idx], labels[idx]
            n = int(len(labels) * 0.8)
            train_x, train_labels = features[:n].to(device), labels[:n].to(device)
            eval_x, eval_labels = features[n:].to(device), labels[n:].to(device)
        else:
            ds_cfg_load = {
                'name': ds_cfg.get('name', self.dataset_name),
                'loader': ds_cfg['loader'],
                'input_dim': ds_cfg['input_dim'],
                'flatten': ds_cfg.get('flatten', True),
            }
            if 'loader_kwargs' in ds_cfg:
                ds_cfg_load['loader_kwargs'] = ds_cfg['loader_kwargs']
            if 'max_samples' in ds_cfg:
                ds_cfg_load['max_samples'] = ds_cfg['max_samples']
            train_x, train_labels, eval_x, eval_labels = load_dataset(ds_cfg_load, device)
            input_dim = train_x.shape[1]
            n_classes = ds_cfg['n_classes']

        self.post({"type": "log",
                    "msg": f"Loaded: {train_x.shape[0]} train, {eval_x.shape[0]} eval"})

        # ── Convert catalog curves to torch ──
        all_curves = torch.tensor(self.catalog_curves, dtype=torch.float32, device=device)
        all_exprs = self.catalog_exprs
        n_catalog = all_curves.shape[0]

        # ── Train each layer ──
        basis_elements = []  # list of (w, b, cidx)
        n_layers = len(self.layer_configs)

        for li, lcfg in enumerate(self.layer_configs):
            if self.stop_event.is_set():
                self.post({"type": "log", "msg": "Stopped by user."})
                break

            layer_num = li + 1
            n_neurons = lcfg['n_neurons']
            mode = lcfg['mode']
            act_idx = lcfg['activation_idx']
            palette_idx = lcfg['palette_indices']

            stag_limit = BASIS_STAG_FIRST if layer_num == 1 else BASIS_STAG_REST
            self.post({"type": "layer_start", "layer": layer_num,
                        "stag_limit": stag_limit})

            # ── Baseline ──
            n_samples = train_x.shape[0]
            with torch.no_grad():
                baselines = []
                for _ in range(10):
                    ix = torch.randint(0, n_samples, (BASIS_BATCH_SIZE,), device=device)
                    bf = basis_forward(train_x[ix], basis_elements, all_curves)
                    acc = knn_fitness(bf.unsqueeze(0), train_labels[ix], n_classes)
                    baselines.append(acc[0].item())
                baseline = float(np.mean(baselines))
            self.post({"type": "baseline", "layer": layer_num, "baseline": baseline})

            # ── Build pool ──
            if mode == "locked":
                if act_idx < 0:
                    self.post({"type": "log", "msg": f"L{layer_num}: no activation, skipping"})
                    continue
                pool_shortlist = [act_idx]
            else:
                # Palette mode: screen user's palette set
                if not palette_idx:
                    self.post({"type": "log", "msg": f"L{layer_num}: empty palette, skipping"})
                    continue
                if len(palette_idx) <= BASIS_SCREEN_TOP_K:
                    pool_shortlist = list(palette_idx)
                else:
                    sorted_idx, scores = screen_activations(
                        palette_idx, all_curves, basis_elements,
                        train_x, train_labels, input_dim, n_classes, n_neurons, device)
                    top_k = min(BASIS_SCREEN_TOP_K, len(palette_idx))
                    pool_shortlist = [palette_idx[sorted_idx[j].item()] for j in range(top_k)]

            n_short = len(pool_shortlist)
            self.post({"type": "log",
                        "msg": f"L{layer_num}: {mode}, pool={n_short}, neurons={n_neurons}"})

            # ── Evolution ──
            P = BASIS_POP_SIZE
            s = 2.0 / (input_dim ** 0.5)
            weights = torch.randn(P, n_neurons, input_dim, device=device) * s
            bias = torch.zeros(P, n_neurons, device=device)

            act_pop = torch.zeros(P, dtype=torch.long, device=device)
            if mode == "locked":
                act_pop[:] = 0  # single activation
            else:
                for i in range(P):
                    act_pop[i] = i % n_short

            bf_val, bi, stag = 0.0, 0, 0
            mr, ms = BASIS_MUT_RATE, BASIS_MUT_SCALE
            gen = 0

            while not self.stop_event.is_set():
                ix = torch.randint(0, n_samples, (BASIS_BATCH_SIZE,), device=device)
                batch = train_x[ix]

                with torch.no_grad():
                    basis_feat = basis_forward(batch, basis_elements, all_curves)
                    linear = torch.einsum('bd,pnd->pbn', batch, weights) + bias.unsqueeze(1)

                    global_act_idx = torch.tensor(
                        [pool_shortlist[act_pop[p].item()] for p in range(P)],
                        dtype=torch.long, device=device)
                    pop_curves = all_curves[global_act_idx]
                    activated = apply_activation_batch(linear, pop_curves)

                    existing_dim = len(basis_elements) * n_neurons if basis_elements else 0
                    if existing_dim > 0:
                        basis_exp = basis_feat.unsqueeze(0).expand(P, -1, -1)
                        combined = torch.cat([basis_exp, activated], dim=2)
                    else:
                        combined = activated

                    fitness = knn_fitness(combined, train_labels[ix], n_classes)

                top = fitness.argmax().item()
                gf = fitness[top].item()

                if gf > bf_val:
                    bf_val, bi, stag = gf, top, 0
                    mr, ms = BASIS_MUT_RATE, BASIS_MUT_SCALE
                else:
                    stag += 1

                # Post progress every 10 gens
                if gen % 10 == 0:
                    best_global = pool_shortlist[act_pop[bi].item()]
                    best_expr = all_exprs[best_global]
                    self.post({
                        "type": "progress",
                        "layer": layer_num,
                        "gen": gen,
                        "fitness": bf_val,
                        "stag": stag,
                        "stag_limit": stag_limit,
                        "expr": best_expr,
                    })

                if stag >= stag_limit:
                    break

                if stag > 0 and stag % 50 == 0:
                    mr = max(mr * 0.92, 0.015)
                    ms = min(ms * 1.12, 0.20)

                # Selection + mutation
                ne = max(1, int(P * BASIS_ELITE_FRAC))
                rk = fitness.argsort(descending=True)
                elites = rk[:ne]
                nc = P - ne

                parents = elites[torch.arange(nc, device=device) % ne]
                nw = weights.clone()
                nb = bias.clone()
                na = act_pop.clone()
                nw[ne:] = weights[parents]
                nb[ne:] = bias[parents]
                na[ne:] = act_pop[parents]

                nw[ne:] += (torch.rand(nc, n_neurons, input_dim, device=device) < mr).float() \
                    * torch.randn(nc, n_neurons, input_dim, device=device) * ms
                nb[ne:] += (torch.rand(nc, n_neurons, device=device) < mr).float() \
                    * torch.randn(nc, n_neurons, device=device) * ms

                # Activation swap (palette mode only)
                if n_short > 1:
                    amut = torch.rand(nc, device=device) < 0.15
                    new_acts = torch.randint(0, n_short, (nc,), device=device)
                    na[ne:] = torch.where(amut, new_acts, na[ne:])

                weights = nw
                bias = nb
                act_pop = na
                gen += 1

            # ── Record result ──
            best_act_local = act_pop[bi].item()
            best_act_global = pool_shortlist[best_act_local]
            best_w = weights[bi].detach().clone()
            best_b = bias[bi].detach().clone()
            best_expr = all_exprs[best_act_global]

            basis_elements.append((best_w, best_b, best_act_global))

            self.post({
                "type": "layer_done",
                "layer": layer_num,
                "baseline": baseline,
                "fitness": bf_val,
                "expr": best_expr,
                "act_idx": best_act_global,
                "gens": gen,
            })
            self.post({"type": "log",
                        "msg": f"L{layer_num} done: {baseline:.1f}% -> {bf_val:.1f}% [{best_expr[:30]}]"})

        # ── Final evaluation ──
        if basis_elements and not self.stop_event.is_set():
            self.post({"type": "status", "msg": "Evaluating..."})
            with torch.no_grad():
                train_feat = basis_forward(train_x, basis_elements, all_curves)
                eval_feat = basis_forward(eval_x, basis_elements, all_curves)
                train_acc = batched_knn_eval(
                    train_feat, train_labels, train_feat, train_labels,
                    n_classes, device, k=KNN_K)
                test_acc = batched_knn_eval(
                    train_feat, train_labels, eval_feat, eval_labels,
                    n_classes, device, k=KNN_K)
            self.post({"type": "done", "train_acc": train_acc, "test_acc": test_acc})
            self.post({"type": "log", "msg": f"Final: train={train_acc:.1f}% test={test_acc:.1f}%"})
        else:
            self.post({"type": "done", "train_acc": None, "test_acc": None})

        # Cleanup GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════════
# 12. NetworkBuilder (main app)
# ═══════════════════════════════════════════════════════════════════════════════

class NetworkBuilder:
    """Main application — drag activations into layers, train networks."""

    def __init__(self, catalog_path):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption("Network Builder")
        self.clock = pygame.time.Clock()
        self.running = True

        self.font = pygame.font.SysFont("consolas", 14)
        self.small_font = pygame.font.SysFont("consolas", 11)

        # Load catalog (with loading screen)
        self.catalog = CatalogData(catalog_path, self.screen, self.font)
        self.curve_renderer = CurveRenderer(self.catalog)

        # Components
        self.palette = ActivationPalette(self.catalog, self.curve_renderer)
        self.stack = NetworkStack(self.catalog, self.curve_renderer)
        self.toolbar = Toolbar()
        self.monitor = TrainingMonitor()
        self.drag = DragController(self.curve_renderer)

        # Training state
        self.training_thread = None
        self.msg_queue = queue.Queue()
        self.is_training = False

    def run(self):
        while self.running:
            self._handle_events()
            self._update()
            self._draw()
            self.clock.tick(FPS_TARGET)

        # Cleanup
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.stop_event.set()
            self.training_thread.join(timeout=5)
        pygame.quit()

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                if self.palette.search_active:
                    self.palette.search_active = False
                    self.palette.search_text = ""
                    self.palette._apply_sort()
                else:
                    self.running = False
                return

            # Close open dropdowns on click elsewhere
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Check if any dropdown is open and click is outside
                if self.toolbar.dataset_dd.is_open:
                    # Let toolbar handle it
                    pass
                if self.palette.sort_dd.is_open:
                    pass
                # Close neuron dropdowns if clicking outside
                for dd in self.stack.neuron_dds:
                    if dd.is_open and not dd.rect.collidepoint(event.pos):
                        n = len(dd.options)
                        list_rect = pygame.Rect(dd.rect.x, dd.rect.bottom,
                                                dd.rect.width, n * dd.item_h)
                        if not list_rect.collidepoint(event.pos):
                            dd.close()

            # Drag handling
            if self.drag.dragging:
                if event.type == pygame.MOUSEMOTION:
                    self.drag.update(event.pos)
                    self.stack.handle_event(event, dragging=True)
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    catalog_idx = self.drag.stop()
                    if catalog_idx >= 0:
                        self.stack.handle_drop(catalog_idx)
                    self.stack.drop_target = -1
                continue

            # Toolbar
            action = self.toolbar.handle_event(event)
            if action == "train":
                self._start_training()
                continue
            elif action == "stop":
                self._stop_training()
                continue
            elif action == "reset":
                self._reset()
                continue
            elif action == "dataset_changed":
                continue
            if self.toolbar.dataset_dd.is_open:
                continue

            # Palette (may initiate drag)
            result = self.palette.handle_event(event)
            if result is not None:
                catalog_idx, pos = result
                self.drag.start(catalog_idx, pos)
                continue

            # Stack
            stack_action = self.stack.handle_event(event, dragging=False)
            if stack_action == "add_empty":
                # Add empty layer with defaults
                self.stack.add_layer(-1, 32, "locked")

    def _update(self):
        """Poll training message queue."""
        while True:
            try:
                msg = self.msg_queue.get_nowait()
            except queue.Empty:
                break

            mtype = msg["type"]
            if mtype == "status":
                self.monitor.status = msg["msg"]
            elif mtype == "log":
                self.monitor.add_log(msg["msg"])
                log(msg["msg"])
            elif mtype == "layer_start":
                self.monitor.current_layer = msg["layer"]
                self.monitor.stag_limit = msg["stag_limit"]
                self.monitor.gen = 0
                self.monitor.stag = 0
                self.monitor.fitness = 0
                self.monitor.best_fitness = 0
                self.monitor.status = f"Training L{msg['layer']}"
            elif mtype == "baseline":
                self.monitor.baseline = msg["baseline"]
            elif mtype == "progress":
                self.monitor.gen = msg["gen"]
                self.monitor.fitness = msg["fitness"]
                self.monitor.best_fitness = max(self.monitor.best_fitness, msg["fitness"])
                self.monitor.stag = msg["stag"]
                self.monitor.stag_limit = msg["stag_limit"]
            elif mtype == "layer_done":
                layer = msg["layer"]
                baseline = msg["baseline"]
                fitness = msg["fitness"]
                expr = msg["expr"]
                act_idx = msg["act_idx"]
                self.monitor.layer_history.append((layer, baseline, fitness, expr))
                # Update the layer slot
                if 0 < layer <= len(self.stack.layers):
                    L = self.stack.layers[layer - 1]
                    L.result_baseline = baseline
                    L.result_fitness = fitness
                    L.result_expr = expr
            elif mtype == "done":
                self.monitor.train_acc = msg.get("train_acc")
                self.monitor.test_acc = msg.get("test_acc")
                self.monitor.status = "Done" if msg.get("train_acc") is not None else "Stopped"
                self.is_training = False
                self.toolbar.set_training(False)
                self.stack.locked = False
            elif mtype == "error":
                self.monitor.status = f"Error: {msg['msg']}"
                self.monitor.add_log(f"ERROR: {msg['msg']}")
                log(f"Training error: {msg['msg']}")
                self.is_training = False
                self.toolbar.set_training(False)
                self.stack.locked = False

    def _draw(self):
        self.screen.fill(BG_COLOR)

        # Main panels
        self.palette.draw(self.screen, self.font, self.small_font)
        self.stack.draw(self.screen, self.font, self.small_font, dragging=self.drag.dragging)
        self.monitor.draw(self.screen, self.font, self.small_font)
        self.toolbar.draw(self.screen, self.font, self.small_font)

        # Status bar
        status_y = WINDOW_H - STATUS_H
        pygame.draw.rect(self.screen, PANEL_BG, (0, status_y, WINDOW_W, STATUS_H))
        pygame.draw.line(self.screen, PANEL_BORDER, (0, status_y), (WINDOW_W, status_y))

        ds_name = self.toolbar.dataset_name
        cfg = DATASET_CONFIGS[ds_name]
        mode = "Training..." if self.is_training else "Ready"
        status_text = f"  {mode}  |  {ds_name} ({cfg['input_dim']}D, {cfg['n_classes']} cls)  |  Layers: {len(self.stack.layers)}"
        st = self.small_font.render(status_text, True, TEXT_DIM)
        self.screen.blit(st, (4, status_y + 6))

        # Overlays (z-order: dropdowns on top)
        self.palette.sort_dd.draw_overlay(self.screen, self.small_font)
        self.stack.draw_overlays(self.screen, self.small_font)
        self.toolbar.draw_overlays(self.screen, self.font)

        # Drag ghost (topmost)
        self.drag.draw(self.screen, self.font)

        pygame.display.flip()

    def _start_training(self):
        if self.is_training:
            return
        if not self.stack.layers:
            self.monitor.add_log("No layers configured!")
            return

        # Check layers have activations
        valid = False
        for L in self.stack.layers:
            if L.activation_idx >= 0 or (L.mode == "palette" and L.palette_indices):
                valid = True
                break
        if not valid:
            self.monitor.add_log("All layers empty. Drag activations first.")
            return

        self.is_training = True
        self.toolbar.set_training(True)
        self.stack.locked = True
        self.monitor.reset()
        self.monitor.status = "Starting..."

        layer_configs = self.stack.get_config()
        dataset_name = self.toolbar.dataset_name

        self.msg_queue = queue.Queue()
        self.training_thread = TrainingThread(
            layer_configs=layer_configs,
            dataset_name=dataset_name,
            catalog_curves=self.catalog.curves,
            catalog_exprs=self.catalog.expressions,
            msg_queue=self.msg_queue,
        )
        self.training_thread.start()
        log(f"Training started: {dataset_name}, {len(layer_configs)} layers")

    def _stop_training(self):
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.stop_event.set()
            self.monitor.status = "Stopping..."

    def _reset(self):
        if self.is_training:
            self._stop_training()
            return
        self.monitor.reset()
        for L in self.stack.layers:
            L.result_baseline = None
            L.result_fitness = None
            L.result_expr = None


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# Headless Mode
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_activation(catalog, spec):
    """Resolve an activation spec to a catalog index.

    spec can be:
      - int: direct catalog index
      - str: exact or substring match on expression (best mean_score wins ties)
    Returns catalog index or -1.
    """
    if isinstance(spec, int):
        if 0 <= spec < catalog.n:
            return spec
        return -1
    if isinstance(spec, str):
        query = spec.lower()
        # Exact match first
        for i, e in enumerate(catalog.expressions_lower):
            if e == query:
                return i
        # Substring match — pick highest mean_score
        matches = []
        for i, e in enumerate(catalog.expressions_lower):
            if query in e:
                matches.append(i)
        if matches:
            best = max(matches, key=lambda i: catalog.mean_scores[i])
            return best
        return -1
    return -1


def run_headless(config, catalog_path, output_path=None):
    """Run training from a JSON config dict, no GUI.

    Config format:
    {
        "dataset": "MNIST",
        "layers": [
            {"activation": "sin(x)", "neurons": 32, "mode": "locked"},
            {"activation": "abs(x)", "neurons": 64, "mode": "locked"},
            {"activation": 1234,     "neurons": 32, "mode": "locked"},
            {
                "mode": "palette",
                "neurons": 32,
                "palette": ["sin(x)", "cos(x)", "abs(x)", 42]
            }
        ]
    }

    - "activation": expression string or catalog index (for locked mode primary,
      also sets primary for palette mode if present)
    - "palette": list of expression strings / catalog indices (palette mode)
    - "neurons": int (default 32)
    - "mode": "locked" | "palette" (default "locked")

    Returns result dict (also written to output_path if given).
    """
    import time as _time
    t0 = _time.time()

    # Load catalog (no pygame)
    catalog = CatalogData(catalog_path)

    # Resolve layers
    dataset_name = config["dataset"]
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                         f"Choose from: {list(DATASET_CONFIGS.keys())}")

    layer_configs = []
    for i, lspec in enumerate(config["layers"]):
        mode = lspec.get("mode", "locked")
        neurons = lspec.get("neurons", 32)

        if mode == "locked":
            act_spec = lspec.get("activation")
            if act_spec is None:
                raise ValueError(f"Layer {i+1}: locked mode requires 'activation'")
            act_idx = resolve_activation(catalog, act_spec)
            if act_idx < 0:
                raise ValueError(f"Layer {i+1}: could not find activation '{act_spec}'")
            log(f"Layer {i+1}: locked [{catalog.expressions[act_idx]}] "
                f"neurons={neurons} (idx={act_idx})")
            layer_configs.append({
                "activation_idx": act_idx,
                "palette_indices": [],
                "mode": "locked",
                "n_neurons": neurons,
            })

        elif mode == "palette":
            palette_specs = lspec.get("palette", [])
            if not palette_specs:
                raise ValueError(f"Layer {i+1}: palette mode requires 'palette' list")
            palette_indices = []
            for ps in palette_specs:
                pidx = resolve_activation(catalog, ps)
                if pidx < 0:
                    log(f"  WARNING: Layer {i+1} palette item '{ps}' not found, skipping")
                    continue
                if pidx not in palette_indices:
                    palette_indices.append(pidx)
            if not palette_indices:
                raise ValueError(f"Layer {i+1}: no valid palette activations resolved")

            # Primary activation
            primary = -1
            act_spec = lspec.get("activation")
            if act_spec is not None:
                primary = resolve_activation(catalog, act_spec)
            if primary < 0:
                primary = palette_indices[0]

            log(f"Layer {i+1}: palette [{len(palette_indices)} acts] "
                f"neurons={neurons} primary=[{catalog.expressions[primary]}]")
            layer_configs.append({
                "activation_idx": primary,
                "palette_indices": palette_indices,
                "mode": "palette",
                "n_neurons": neurons,
            })
        else:
            raise ValueError(f"Layer {i+1}: unknown mode '{mode}'")

    # Run training synchronously (blocking, same thread)
    msg_queue = queue.Queue()
    thread = TrainingThread(
        layer_configs=layer_configs,
        dataset_name=dataset_name,
        catalog_curves=catalog.curves,
        catalog_exprs=catalog.expressions,
        msg_queue=msg_queue,
    )

    log(f"Headless training: {dataset_name}, {len(layer_configs)} layers")
    thread.start()
    thread.join()

    # Collect results from queue
    layer_results = []
    train_acc = None
    test_acc = None
    errors = []

    while True:
        try:
            msg = msg_queue.get_nowait()
        except queue.Empty:
            break
        mtype = msg["type"]
        if mtype == "layer_done":
            layer_results.append({
                "layer": msg["layer"],
                "baseline": msg["baseline"],
                "fitness": msg["fitness"],
                "expression": msg["expr"],
                "activation_idx": msg["act_idx"],
                "generations": msg["gens"],
            })
        elif mtype == "done":
            train_acc = msg.get("train_acc")
            test_acc = msg.get("test_acc")
        elif mtype == "error":
            errors.append(msg["msg"])
        elif mtype == "log":
            log(msg["msg"])

    elapsed = _time.time() - t0
    result = {
        "dataset": dataset_name,
        "n_layers": len(layer_configs),
        "layer_results": layer_results,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "errors": errors,
        "elapsed_s": round(elapsed, 1),
    }

    log(f"Headless done in {elapsed:.1f}s: "
        f"train={train_acc}% test={test_acc}%")

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        log(f"Results written to {output_path}")

    return result


def find_catalog():
    """Search for master_catalog.json in common locations."""
    candidates = [
        Path(__file__).parent / "results" / "master_catalog.json",
        Path(__file__).parent / "master_catalog.json",
    ]
    # Also search results subdirs
    results_dir = Path(__file__).parent / "results"
    if results_dir.exists():
        for d in sorted(results_dir.iterdir(), reverse=True):
            if d.is_dir() and "master_catalog" in d.name:
                p = d / "master_catalog.json"
                if p.exists():
                    candidates.insert(0, p)
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Network Builder — GUI or headless training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Headless mode examples:

  python network_builder.py --headless --config run.json
  python network_builder.py --headless --config run.json --output results.json

  run.json format:
  {
    "dataset": "MNIST",
    "layers": [
      {"activation": "sin(x)", "neurons": 32, "mode": "locked"},
      {"activation": "abs(x)", "neurons": 64},
      {"mode": "palette", "neurons": 32, "palette": ["sin(x)", "cos(x)", "x"]}
    ]
  }
""")
    parser.add_argument("--catalog", type=str, default=None,
                        help="Path to master_catalog.json")
    parser.add_argument("--headless", action="store_true",
                        help="Run without GUI, requires --config")
    parser.add_argument("--config", type=str, default=None,
                        help="JSON config file for headless mode")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output JSON path for headless results")
    args = parser.parse_args()

    catalog_path = args.catalog or find_catalog()
    if catalog_path is None:
        print("ERROR: Could not find master_catalog.json")
        print("Run with --catalog path/to/master_catalog.json")
        sys.exit(1)

    log(f"Using catalog: {catalog_path}")

    if args.headless:
        if not args.config:
            parser.error("--headless requires --config <path.json>")
        with open(args.config, encoding='utf-8') as f:
            config = json.load(f)
        result = run_headless(config, catalog_path, args.output)
        # Print summary to stdout
        print(json.dumps(result, indent=2))
    else:
        app = NetworkBuilder(catalog_path)
        app.run()


if __name__ == "__main__":
    main()
