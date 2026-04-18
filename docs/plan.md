# Optimizer — Design Plan

## Goal

A standalone Python package that fits a rigged skeleton to an arbitrary mesh
sequence using differentiable Forward Kinematics (FK) + Linear Blend Skinning
(LBS) via gradient descent (Adam).

**Inputs**
- `rig.npz` — a rig file containing:
  - `joints`  — `(J, 3)` float32, rest-pose joint positions in world space
  - `skin`    — `(V, J)` float32, per-vertex skinning weights
  - `parents` — `(J,)` object array, parent index per joint (`None` = root)
  - `names`   — `(J,)` str array, joint names (optional, not used by the fitting logic)
- `v0` — `(V, 3)` float32, rest-pose vertices the rig was built from (kept separate from rig.npz)
- Mesh sequence — `(T, V, 3)` float32 `.npy` file, vertex positions across T frames

**Input example**

- `rig.npz`: rig for an animal with 24 joints and 6890 vertices
  - `joints` — shape `(24, 3)`, xyz position of each joint
  - `skin` — shape `(6890, 24)`, skinning weights of each vertex over 24 joints
  - `parents` — `[None, 0, 1, 2, ...]`, root is `None`, others are parent joint indices
  - `names` — `['root', 'spine', 'chest', ...]` (optional)
- Mesh sequence `verts.npy`: shape `(120, 6890, 3)`, 120-frame animation, xyz coordinates per vertex per frame

**Output**

`fit_sequence` returns a numpy array:

- **Shape**: `(T, V, 3)`
  - `T`: number of frames, same as the input sequence
  - `V`: number of vertices, same as the input sequence
  - `3`: world-space xyz coordinates per vertex
- **dtype**: `float32`
- **Coordinate space**: world space, consistent with the input mesh sequence
- **All T frames**: vertex positions produced by FK + LBS fitting; each frame stores the prediction at the lowest-loss iteration across `n_iters` steps

**Output example**

Given a 120-frame sequence with 6890 vertices:

- shape `(120, 6890, 3)`, dtype `float32`
- `fitted[0]` — shape `(6890, 3)`, fitted xyz positions for frame 0 produced by FK + LBS
- `fitted[1]` — shape `(6890, 3)`, fitted xyz positions for frame 1 produced by FK + LBS
- `fitted[t, v, :]` — `[x, y, z]` of vertex `v` at frame `t`
- Saved as a `.npy` file, reloadable with `np.load`

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
├── run.py               ← single-sequence CLI
└── batch.py             ← batch processing (reconstruction / transfer)
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
    names:   np.ndarray | None  # (J,) joint names, optional

def load_rig(npz_path) -> Rig
```

### `optimizer/fit.py`

```python
def fit_sequence(
    rig:       Rig,
    verts_seq: np.ndarray,   # (T, V, 3)
    v0:        np.ndarray,   # (V, 3) rest-pose vertices
    n_iters:   int   = 500,
    lr:        float = 1e-2,
    device:    str   = 'cuda',
    verbose:   bool  = True,
) -> np.ndarray              # (T, V, 3) float32
```

- If frame 0 is identical to `v0`, it is copied directly and skipped; otherwise it is fitted along with all other frames.
- All fitted frames are warm-started from the previous frame's `axis_angle` / `root_trans` (the first fitted frame is initialised to zero).
- Loss: mean L2 distance between predicted and target vertices.
- Optimizer: Adam.
- Returns the frame with the lowest loss seen during the n_iters iterations
  (best-loss tracking).

### `run.py` (single sequence)

Does one thing: given a rig, rest-pose vertices (v0), and a mesh sequence, output a fitted sequence. No knowledge of HDF5, directory structure, or sampling logic.

```
python run.py --rig rig.npz --verts verts.npy --v0 v0.npy --output fitted.npy
```

Arguments: `--rig`, `--verts`, `--v0`, `--output`, `--n_iters`, `--lr`, `--device`.

### `batch.py` (batch processing)

Handles HDF5 loading, directory management, sequence sampling, and loops over all rigs.

```
# Training Reconstruction
python batch.py --mode reconstruction \
  --rig_dir rigs --hdf5 dt4d.hdf5 --output_dir results/output

# Cross-Motion Transfer (randomly samples another sequence from the same animal)
python batch.py --mode transfer \
  --rig_dir rigs --hdf5 dt4d.hdf5 --output_dir results/output
```

Arguments: `--mode reconstruction / transfer`, `--rig_dir`, `--hdf5`, `--output_dir`, `--n_iters`, `--lr`, `--device`, `--seed`. Already-processed sequences are skipped automatically.

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

## Batch Mode

The rig directory structure mirrors the HDF5 layout exactly — no string parsing needed:

```
rigs/
├── bear84Q/
│   └── bear84Q_Landing/
│       └── rig.npz
└── deerOMG/
    └── deerOMG_death2/
        └── rig.npz
```

**Training Reconstruction** (same-sequence fitting)
- Each `rig.npz` fits the sequence it was built from
- `rigs/bear84Q/bear84Q_Landing/rig.npz` → fits `bear84Q/bear84Q_Landing` in HDF5

**Cross-Motion Transfer** (cross-action fitting)
- Each `rig.npz` randomly samples another sequence from the same animal
- `rigs/bear84Q/bear84Q_Landing/rig.npz` → randomly picks `bear84Q/bear84Q_AttackrunRM`

### Output directory structure

```
output_dir/
├── reconstruction/
│   └── bear84Q/
│       └── bear84Q_Landing/
│           └── fitted_vertices.npy
└── transfer/
    └── bear84Q/
        ├── fitted_vertices.npy
        └── meta.json
```

`meta.json` records which rig and target sequence were used:
```json
{"rig": "bear84Q_Landing", "target": "bear84Q_AttackrunRM"}
```

### CLI usage

```
# Training Reconstruction
python batch.py --mode reconstruction \
  --rig_dir rigs --hdf5 dt4d.hdf5 --output_dir results/output

# Cross-Motion Transfer
python batch.py --mode transfer \
  --rig_dir rigs --hdf5 dt4d.hdf5 --output_dir results/output
```

Arguments: `--mode reconstruction / transfer`, `--rig_dir`, `--hdf5`, `--output_dir`, `--n_iters`, `--lr`, `--device`, `--seed`. Already-processed sequences are skipped automatically.

---

## Dependencies

- `torch` (≥ 2.0) — differentiable FK/LBS, Adam
- `numpy`
- `h5py` — loading mesh sequences from `dt4d.hdf5`
- `tqdm` — progress bars
