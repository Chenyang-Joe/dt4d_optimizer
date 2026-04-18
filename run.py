#!/usr/bin/env python3
"""
Fit a single rig to a mesh sequence using FK + LBS + Adam.

Inputs:
  --rig    rig.npz          skeleton file (joints, skinning weights, parent indices)
  --verts  verts.npy        target mesh sequence, shape (T, V, 3) float32
  --v0     v0.npy           rest-pose vertices the rig was built from, shape (V, 3) float32

Output:
  --output fitted.npy       fitted vertex sequence, shape (T, V, 3) float32

Usage:
  python run.py --rig rig.npz --verts verts.npy --v0 v0.npy --output fitted.npy
"""

import argparse
from pathlib import Path

import numpy as np

from optimizer import load_rig, fit_sequence


def main():
    parser = argparse.ArgumentParser(description='FK+LBS mesh sequence fitting')
    parser.add_argument('--rig',     required=True,              help='Path to rig.npz')
    parser.add_argument('--verts',   required=True,              help='Path to input mesh sequence .npy, shape (T, V, 3)')
    parser.add_argument('--v0',      required=True,              help='Path to rest-pose vertices .npy, shape (V, 3)')
    parser.add_argument('--output',  required=True,              help='Path to output fitted sequence .npy, shape (T, V, 3)')
    parser.add_argument('--n_iters', type=int,   default=500,    help='Adam iterations per frame')
    parser.add_argument('--lr',      type=float, default=1e-2,   help='Adam learning rate')
    parser.add_argument('--device',  type=str,   default='cuda', help='torch device')
    args = parser.parse_args()

    rig       = load_rig(args.rig)
    verts_seq = np.load(args.verts).astype(np.float32)
    v0        = np.load(args.v0).astype(np.float32)

    fitted = fit_sequence(
        rig,
        verts_seq,
        v0=v0,
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
