from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Rig:
    joints:  np.ndarray        # (J, 3) float32 — rest-pose joint positions (world space)
    weights: np.ndarray        # (V, J) float32 — per-vertex skinning weights
    parents: list              # length J; None for root joint, int index otherwise
    names:   np.ndarray | None = None  # (J,) str — joint names, optional


def _parse_parents(parents_raw) -> list:
    parents = []
    for p in parents_raw:
        if p is None or (hasattr(p, '__int__') and int(p) < 0):
            parents.append(None)
        else:
            parents.append(int(p))
    return parents


def load_rig(npz_path: str | Path) -> Rig:
    """Load rig from a .npz file with required keys: joints, skin, parents; optional: names."""
    data = np.load(str(npz_path), allow_pickle=True)
    return Rig(
        joints  = data['joints'].astype(np.float32),
        weights = data['skin'].astype(np.float32),
        parents = _parse_parents(data['parents']),
        names   = data['names'] if 'names' in data else None,
    )
