from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from .rig import Rig
from .transforms import axis_angle_to_matrix, forward_kinematics, lbs


def _fit_frame(
    v0:         torch.Tensor,        # (V, 3) rest-pose vertices
    W:          torch.Tensor,        # (V, J) skinning weights
    joints:     torch.Tensor,        # (J, 3) rest-pose joint positions
    parents:    list,
    target:     torch.Tensor,        # (V, 3) target vertices for this frame
    n_iters:    int,
    lr:         float,
    axis_angle: torch.Tensor | None, # warm-start from previous frame
    root_trans: torch.Tensor | None,
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """
    Fit one frame via Adam. Returns (best_verts, axis_angle, root_trans).
    best_verts is the prediction at the iteration with the lowest loss.
    """
    J      = joints.shape[0]
    device = v0.device

    if axis_angle is not None:
        aa = axis_angle.detach().clone().requires_grad_(True)
        rt = root_trans.detach().clone().requires_grad_(True)
    else:
        aa = torch.zeros(J, 3, device=device, requires_grad=True)
        rt = torch.zeros(3,   device=device, requires_grad=True)

    opt        = torch.optim.Adam([aa, rt], lr=lr)
    best_loss  = float('inf')
    best_verts = None

    for _ in range(n_iters):
        opt.zero_grad()
        R_mats = axis_angle_to_matrix(aa)
        T_mats = forward_kinematics(joints, parents, R_mats, rt)
        v_pred = lbs(v0, W, T_mats)
        loss   = ((v_pred - target) ** 2).mean()
        loss.backward()
        opt.step()

        if loss.item() < best_loss:
            best_loss  = loss.item()
            best_verts = v_pred.detach().cpu().numpy()

    return best_verts, aa, rt


def fit_sequence(
    rig:       Rig,
    verts_seq: np.ndarray,   # (T, V, 3) float32 — target mesh sequence
    v0:        np.ndarray,   # (V, 3)    float32 — rest-pose vertices the rig was built from
    n_iters:   int   = 500,
    lr:        float = 1e-2,
    device:    str   = 'cuda',
    verbose:   bool  = True,
) -> np.ndarray:             # (T, V, 3) float32
    """
    Fit the rig to every frame of verts_seq using FK + LBS + Adam.

    v0 is the rest-pose mesh the rig was built from (kept separate from rig.npz).
    All T frames are fitted. Returns a (T, V, 3) float32 array.
    """
    T, V, _ = verts_seq.shape

    joints  = torch.tensor(rig.joints,  device=device)
    W       = torch.tensor(rig.weights, device=device)
    v0_t    = torch.tensor(v0,          device=device)

    out     = np.zeros((T, V, 3), dtype=np.float32)
    prev_aa = None
    prev_rt = None

    # if frame 0 is identical to the rest pose, skip fitting and copy directly
    frame0_is_rest = (verts_seq[0] is v0) or np.array_equal(verts_seq[0], v0)
    start = 0
    if frame0_is_rest:
        out[0] = v0
        start  = 1
        if verbose:
            print('frame 0 == rest pose, skipping fit')

    frames = range(start, T)
    if verbose:
        frames = tqdm(frames, desc='fitting frames')

    for t in frames:
        target = torch.tensor(verts_seq[t], device=device)
        best_verts, prev_aa, prev_rt = _fit_frame(
            v0_t, W, joints, rig.parents, target,
            n_iters, lr, prev_aa, prev_rt,
        )
        out[t] = best_verts

    return out
