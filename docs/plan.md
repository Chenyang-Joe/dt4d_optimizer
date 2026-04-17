# Optimizer — Design Plan

## Goal

A standalone Python package that fits a rigged skeleton to an arbitrary mesh
sequence using differentiable Forward Kinematics (FK) + Linear Blend Skinning
(LBS) via gradient descent (Adam).

**Inputs**
- `rig.npz` — a rig file produced by UniRig_on_dt4d's Stage 1 pipeline,
  containing:
  - `joints`  — `(J, 3)` float32, rest-pose joint positions in world space
  - `skin`    — `(V, J)` float32, per-vertex skinning weights
  - `parents` — `(J,)` object array, parent index per joint (`None` = root)
  - `names`   — `(J,)` str array, joint names
- Mesh sequence — `(T, V, 3)` float32 array of vertex positions across T frames.
  Can be supplied as:
  - a `.npy` file
  - a key inside `dt4d.hdf5` (HDF5 group `<animal>/<seq_name>/vertices`)

**Output**
- `(T, V, 3)` float32 numpy array of **fitted** vertex positions.
  - Frame 0: rest pose, returned as-is (`verts_seq[0]`).
  - Frames 1…T-1: result of per-frame gradient-descent fitting.

---

## Repository Layout

```
optimizer/
├── docs/
│   └── plan.md          ← this file
├── optimizer/
│   ├── __init__.py      ← public API: Rig, load_rig, fit_sequence
│   ├── transforms.py    ← pure differentiable math (no I/O)
│   ├── rig.py           ← Rig dataclass + load_rig()
│   └── fit.py           ← fit_frame(), fit_sequence()
└── run.py               ← CLI entry point
```

---

## Module Responsibilities

### `optimizer/transforms.py`

Pure PyTorch math, no I/O, no data loading.

| Function | Signature | Description |
|---|---|---|
| `axis_angle_to_matrix` | `(J,3) → (J,3,3)` | Rodrigues' rotation formula |
| `forward_kinematics` | `(joints, parents, R, root_trans) → (J,4,4)` | Propagates local rotations root→leaves; returns per-joint skinning matrices |
| `lbs` | `(v0, W, T) → (V,3)` | Linear Blend Skinning blend |

### `optimizer/rig.py`

```python
@dataclass
class Rig:
    joints:  np.ndarray   # (J, 3) rest-pose joint positions
    weights: np.ndarray   # (V, J) skinning weights
    parents: list         # length J; None for root, int otherwise
    names:   np.ndarray   # (J,) joint names

def load_rig(npz_path) -> Rig
```

### `optimizer/fit.py`

```python
def fit_sequence(
    rig:       Rig,
    verts_seq: np.ndarray,   # (T, V, 3)
    n_iters:   int   = 500,
    lr:        float = 1e-2,
    device:    str   = 'cuda',
    verbose:   bool  = True,
) -> np.ndarray              # (T, V, 3) float32
```

- Frame 0 → rest pose, copied directly.
- Frames 1…T-1 → gradient descent, warm-started from previous frame's
  `axis_angle` / `root_trans`.
- Loss: mean L2 distance between predicted and target vertices.
- Optimizer: Adam.
- Returns the frame with the lowest loss seen during the n_iters iterations
  (best-loss tracking).

### `run.py` (CLI)

```
# From HDF5
python run.py --rig rig.npz --hdf5 dt4d.hdf5 --key animal/seq_name --output out.npy

# From pre-saved numpy array
python run.py --rig rig.npz --verts verts.npy --output out.npy
```

Arguments: `--rig`, `--output`, `--n_iters`, `--lr`, `--device`,
mutually-exclusive source group `--hdf5 / --verts` (with `--key` for HDF5).

---

## Fitting Algorithm

For each frame t = 1 … T-1:

1. Warm-start: initialise `axis_angle` and `root_trans` from the previous
   frame's solution (or zeros for t=1).
2. Run `n_iters` steps of Adam:
   - `R = axis_angle_to_matrix(axis_angle)`          → `(J, 3, 3)`
   - `T_mats = forward_kinematics(joints, parents, R, root_trans)` → `(J, 4, 4)`
   - `v_pred = lbs(v0, W, T_mats)`                   → `(V, 3)`
   - `loss = mean((v_pred - target)²)`
   - `loss.backward(); optimizer.step()`
3. Track `best_verts` (lowest loss seen).
4. Store `best_verts` in output array at index t.

---

## Dependencies

- `torch` (≥ 2.0) — differentiable FK/LBS, Adam
- `numpy`
- `h5py` — loading mesh sequences from `dt4d.hdf5`
- `tqdm` — progress bars
