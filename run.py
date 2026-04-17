#!/usr/bin/env python3
"""
CLI for the optimizer package.

Usage — load mesh sequence from HDF5:
  python run.py --rig rig.npz --hdf5 dt4d.hdf5 --key animal/seq_name --output out.npy

Usage — load mesh sequence from a .npy array (T, V, 3):
  python run.py --rig rig.npz --verts verts.npy --output out.npy
"""

import argparse
from pathlib import Path

import numpy as np

from optimizer import load_rig, fit_sequence


def main():
    parser = argparse.ArgumentParser(description='FK+LBS mesh sequence fitting')
    parser.add_argument('--rig',     required=True,               help='Path to rig.npz')
    parser.add_argument('--output',  required=True,               help='Output .npy path, shape (T, V, 3)')
    parser.add_argument('--n_iters', type=int,   default=500,     help='Adam iterations per frame')
    parser.add_argument('--lr',      type=float, default=1e-2,    help='Adam learning rate')
    parser.add_argument('--device',  type=str,   default='cuda',  help='torch device')

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--hdf5',  help='Path to dt4d.hdf5')
    src.add_argument('--verts', help='Path to .npy array of shape (T, V, 3)')
    parser.add_argument('--key', help='HDF5 dataset key, e.g. animal/seq_name (required with --hdf5)')

    args = parser.parse_args()

    rig = load_rig(args.rig)

    if args.hdf5:
        if not args.key:
            parser.error('--key is required when using --hdf5')
        import h5py
        with h5py.File(args.hdf5, 'r') as f:
            verts_seq = f[args.key]['vertices'][:].astype(np.float32)
    else:
        verts_seq = np.load(args.verts).astype(np.float32)

    fitted = fit_sequence(
        rig,
        verts_seq,
        n_iters=args.n_iters,
        lr=args.lr,
        device=args.device,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), fitted)
    print(f'Saved: {out_path}  shape={fitted.shape}  dtype={fitted.dtype}')


if __name__ == '__main__':
    main()
