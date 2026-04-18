#!/usr/bin/env python3
"""
Batch fitting over a directory of rigs and a DT4D HDF5 dataset.

Two modes:

  reconstruction  — each rig fits its own source sequence
  transfer        — each rig fits a randomly sampled sequence from the same animal

Rig directory structure must mirror the HDF5 layout:
  rigs/
  ├── bear84Q/
  │   └── bear84Q_Landing/
  │       └── rig.npz
  └── deerOMG/
      └── deerOMG_death2/
          └── rig.npz

Usage:
  python batch.py --mode reconstruction --rig_dir rigs --hdf5 dt4d.hdf5 --output_dir output
  python batch.py --mode transfer       --rig_dir rigs --hdf5 dt4d.hdf5 --output_dir output

Output structure:
  output_dir/
  ├── reconstruction/
  │   └── bear84Q/
  │       └── bear84Q_Landing/
  │           └── fitted_vertices.npy
  └── transfer/
      └── bear84Q/
          └── fitted_vertices.npy
"""

import argparse
import json
import random
from pathlib import Path

import h5py
import numpy as np

from optimizer import load_rig, fit_sequence


def get_animal_keys(hdf5_path: str, animal: str) -> list[str]:
    """Return all sequence keys for a given animal in the HDF5 file."""
    with h5py.File(hdf5_path, 'r') as f:
        if animal not in f:
            return []
        return [f'{animal}/{seq}' for seq in f[animal].keys()]


def load_verts(hdf5_path: str, key: str) -> np.ndarray:
    with h5py.File(hdf5_path, 'r') as f:
        return f[key]['vertices'][:].astype(np.float32)


def run_batch(
    mode:       str,
    rig_dir:    str,
    hdf5_path:  str,
    output_dir: str,
    n_iters:    int,
    lr:         float,
    device:     str,
    seed:       int,
):
    random.seed(seed)
    rig_dir    = Path(rig_dir)
    output_dir = Path(output_dir) / mode

    # find all rig.npz files under rigs/<animal>/<seq>/rig.npz
    rig_paths = sorted(rig_dir.glob('**/rig.npz'))
    print(f'Found {len(rig_paths)} rigs under {rig_dir}')

    for rig_path in rig_paths:
        seq_dir  = rig_path.parent
        seq_name = seq_dir.name            # e.g. bear84Q_Landing
        animal   = seq_dir.parent.name     # e.g. bear84Q
        rig_key  = f'{animal}/{seq_name}'  # e.g. bear84Q/bear84Q_Landing

        if mode == 'reconstruction':
            target_key = rig_key
            out_subdir = output_dir / animal / seq_name

        elif mode == 'transfer':
            all_keys = get_animal_keys(hdf5_path, animal)
            others   = [k for k in all_keys if k != rig_key]
            if not others:
                print(f'  [skip] no other sequences found for {animal}')
                continue
            target_key = random.choice(others)
            out_subdir = output_dir / animal

        out_path = out_subdir / 'fitted_vertices.npy'
        if out_path.exists():
            print(f'  [skip] {out_path}')
            continue

        print(f'  {rig_key}  ->  {target_key}')

        rig       = load_rig(rig_path)
        v0        = load_verts(hdf5_path, rig_key)[0]   # frame 0 of rig's source sequence
        verts_seq = load_verts(hdf5_path, target_key)

        fitted = fit_sequence(rig, verts_seq, v0=v0, n_iters=n_iters, lr=lr, device=device)

        out_subdir.mkdir(parents=True, exist_ok=True)
        np.save(str(out_path), fitted)
        print(f'    saved {out_path}  shape={fitted.shape}')

        if mode == 'transfer':
            meta = {'rig': seq_name, 'target': target_key.split('/')[-1]}
            (out_subdir / 'meta.json').write_text(json.dumps(meta, indent=2))


def main():
    parser = argparse.ArgumentParser(description='Batch FK+LBS fitting')
    parser.add_argument('--mode',       required=True, choices=['reconstruction', 'transfer'])
    parser.add_argument('--rig_dir',    required=True, help='Root rigs directory (rigs/<animal>/<seq>/rig.npz)')
    parser.add_argument('--hdf5',       required=True, help='Path to dt4d.hdf5')
    parser.add_argument('--output_dir', required=True, help='Root output directory')
    parser.add_argument('--n_iters',    type=int,   default=500)
    parser.add_argument('--lr',         type=float, default=1e-2)
    parser.add_argument('--device',     type=str,   default='cuda')
    parser.add_argument('--seed',       type=int,   default=42,  help='Random seed for transfer sampling')
    args = parser.parse_args()

    run_batch(
        mode       = args.mode,
        rig_dir    = args.rig_dir,
        hdf5_path  = args.hdf5,
        output_dir = args.output_dir,
        n_iters    = args.n_iters,
        lr         = args.lr,
        device     = args.device,
        seed       = args.seed,
    )


if __name__ == '__main__':
    main()
