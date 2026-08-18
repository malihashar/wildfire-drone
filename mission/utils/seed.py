"""Reproducibility helpers for research experiments."""

from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int | None) -> np.random.Generator:
    """
    Seed Python and NumPy RNGs.

    Returns a NumPy Generator for use in simulation code. Passing ``None``
    leaves global RNGs untouched and returns an unseeded Generator.
    """
    if seed is None:
        return np.random.default_rng()

    random.seed(seed)
    return np.random.default_rng(seed)
