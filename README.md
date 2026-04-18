# dt4d_optimizer

Fits a rigged skeleton to a DT4D mesh sequence using differentiable FK + LBS + Adam.

## Usage

**Single sequence**
```bash
python run.py \
  --rig    rig.npz \
  --verts  verts.npy \
  --v0     v0.npy \
  --output fitted.npy
```

**Batch (reconstruction)**
```bash
python batch.py --mode reconstruction \
  --rig_dir input/rigs \
  --hdf5    /path/to/dt4d.hdf5 \
  --output_dir output
```

**Batch (transfer)**
```bash
python batch.py --mode transfer \
  --rig_dir input/rigs \
  --hdf5    /path/to/dt4d.hdf5 \
  --output_dir output
```

## Inputs

| File | Shape | Description |
|---|---|---|
| `rig.npz` | — | Skeleton: `joints (J,3)`, `skin (V,J)`, `parents (J,)` |
| `v0.npy` | `(V, 3)` | Rest-pose vertices the rig was built from |
| `verts.npy` | `(T, V, 3)` | Target mesh sequence |

## Output

`fitted_vertices.npy` — `(T, V, 3)` float32, world-space fitted vertices per frame.

## Rig directory structure

Must mirror the HDF5 layout:
```
input/rigs/
├── bear84Q/
│   └── bear84Q_Landing/
│       └── rig.npz
└── deerOMG/
    └── deerOMG_death2/
        └── rig.npz
```

## Output directory structure

```
output/
├── reconstruction/
│   └── <animal>/<seq>/fitted_vertices.npy
└── transfer/
    └── <animal>/
        ├── fitted_vertices.npy
        └── meta.json
```

## Dependencies

`torch >= 2.0`, `numpy`, `h5py`, `tqdm`
