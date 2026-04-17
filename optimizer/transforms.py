import torch


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Rodrigues' formula: (J, 3) axis-angle -> (J, 3, 3) rotation matrices.
    Direction = rotation axis; magnitude = angle theta.
    """
    angle = torch.norm(axis_angle, dim=-1, keepdim=True).clamp(min=1e-8)
    axis  = axis_angle / angle

    cos_a = torch.cos(angle).unsqueeze(-1)   # (J, 1, 1)
    sin_a = torch.sin(angle).unsqueeze(-1)

    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    zeros   = torch.zeros_like(x)

    K = torch.stack([
        zeros, -z,  y,
        z,  zeros, -x,
        -y,  x,  zeros,
    ], dim=-1).reshape(-1, 3, 3)

    I = torch.eye(3, device=axis_angle.device).unsqueeze(0)
    return cos_a * I + sin_a * K + (1 - cos_a) * torch.bmm(
        axis.unsqueeze(-1), axis.unsqueeze(-2)
    )


def forward_kinematics(
    joints:     torch.Tensor,  # (J, 3)  rest-pose joint positions
    parents:    list,          # length J; None for root, int index otherwise
    R:          torch.Tensor,  # (J, 3, 3) local rotation per joint
    root_trans: torch.Tensor,  # (3,) global root translation
) -> torch.Tensor:             # (J, 4, 4) per-joint skinning matrices
    """
    Propagate local rotations root->leaves.
      global_R[j]   = global_R[parent] @ R[j]
      global_pos[j] = global_pos[parent] + global_R[parent] @ (joints[j] - joints[parent])
    Skinning matrix T[j]:
      T[j, :3, :3] = global_R[j]
      T[j, :3,  3] = global_pos[j] - global_R[j] @ joints[j]
    """
    J      = joints.shape[0]
    device = joints.device

    global_R   = [None] * J
    global_pos = [None] * J

    for j in range(J):
        p = parents[j]
        if p is None:
            global_R[j]   = R[j]
            global_pos[j] = joints[j] + root_trans
        else:
            global_R[j]   = global_R[p] @ R[j]
            global_pos[j] = global_pos[p] + global_R[p] @ (joints[j] - joints[p])

    T = torch.zeros(J, 4, 4, device=device)
    for j in range(J):
        T[j, :3, :3] = global_R[j]
        T[j, :3,  3] = global_pos[j] - global_R[j] @ joints[j]
        T[j,  3,  3] = 1.0
    return T


def lbs(
    v0: torch.Tensor,  # (V, 3)    rest-pose vertices
    W:  torch.Tensor,  # (V, J)    skinning weights (each row sums to 1)
    T:  torch.Tensor,  # (J, 4, 4) per-joint skinning matrices from FK
) -> torch.Tensor:     # (V, 3)    posed vertices
    """
    Linear Blend Skinning: v_pred[i] = sum_j  W[i,j] * T[j] @ v0_hom[i]
    """
    V = v0.shape[0]
    v0_hom     = torch.cat([v0, torch.ones(V, 1, device=v0.device)], dim=-1)
    Tv         = torch.einsum('jab,vb->jva', T, v0_hom)
    v_pred_hom = torch.einsum('vj,jva->va', W, Tv)
    return v_pred_hom[:, :3]
