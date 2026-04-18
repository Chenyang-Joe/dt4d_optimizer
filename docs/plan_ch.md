# Optimizer — 设计方案

## 目标

一个独立的 Python 包，使用可微分的前向运动学（FK）+ 线性混合蒙皮（LBS）通过梯度下降（Adam）将带绑定的骨骼拟合到任意网格序列。

**输入**
- `rig.npz` — 绑定文件，包含：
  - `joints`  — `(J, 3)` float32，世界坐标系下的静置姿态关节位置
  - `skin`    — `(V, J)` float32，每顶点蒙皮权重
  - `parents` — `(J,)` object 数组，每个关节的父节点索引（`None` = 根节点）
  - `names`   — `(J,)` 字符串数组，关节名称（可选）
- `v0` — `(V, 3)` float32，生成 rig 时的 rest pose 顶点（独立于 rig.npz 存储，遵循标准做法）
- 网格序列 — `(T, V, 3)` float32 的 `.npy` 文件，包含 T 帧的顶点位置

**输入示例**

- `rig.npz`：24 个关节、6890 个顶点的动物绑定
  - `joints` — shape `(24, 3)`，24 个关节的 xyz 坐标
  - `skin` — shape `(6890, 24)`，6890 个顶点各自对 24 个关节的权重
  - `parents` — `[None, 0, 1, 2, ...]`，根节点为 `None`，其余为父节点编号
  - `names` — `['root', 'spine', 'chest', ...]`（可选）
- 网格序列 `verts.npy`：shape `(120, 6890, 3)`，120 帧动画，每帧 6890 个顶点的 xyz 坐标

**输出**

`fit_sequence` 返回一个 numpy 数组：

- **形状**：`(T, V, 3)`
  - `T`：帧数，与输入网格序列相同
  - `V`：顶点数，与输入网格序列相同
  - `3`：每个顶点的世界坐标系 xyz 坐标
- **数据类型**：`float32`
- **坐标空间**：与输入网格序列一致的世界坐标系
- **所有 T 帧**：均为 FK + LBS 拟合结果；每帧取 `n_iters` 次迭代中损失最低时的预测值

**输出示例**

输入为 120 帧、6890 个顶点的序列时：

- shape `(120, 6890, 3)`，dtype `float32`
- `fitted[0]` — shape `(6890, 3)`，第 0 帧经 FK + LBS 拟合后的顶点 xyz（世界坐标）
- `fitted[1]` — shape `(6890, 3)`，第 1 帧经 FK + LBS 拟合后的顶点 xyz（世界坐标）
- `fitted[t, v, :]` — 第 `t` 帧第 `v` 个顶点的 `[x, y, z]`
- 保存为 `.npy` 文件，用 `np.load` 直接读回

---

## 仓库结构

```
optimizer/
├── docs/
│   └── plan.md          ← 英文版方案
│   └── plan_ch.md       ← 本文件
├── optimizer/
│   ├── __init__.py      ← 公开 API：Rig、load_rig、fit_sequence
│   ├── transforms.py    ← 纯可微分数学运算（无 I/O）
│   ├── rig.py           ← Rig 数据类 + load_rig()
│   └── fit.py           ← fit_frame()、fit_sequence()
├── run.py               ← 单条序列命令行入口
└── batch.py             ← 批量处理（reconstruction / transfer）
```

---

## 各模块职责

### `optimizer/transforms.py`

纯 PyTorch 数学运算，无 I/O，无数据加载。

| 函数 | 签名 | 说明 |
|---|---|---|
| `axis_angle_to_matrix` | `(J,3) → (J,3,3)` | Rodrigues 旋转公式，将轴角向量转为旋转矩阵 |
| `forward_kinematics` | `(joints, parents, R, root_trans) → (J,4,4)` | 从根节点向叶节点传播局部旋转，返回每个关节的蒙皮矩阵 |
| `lbs` | `(v0, W, T) → (V,3)` | 线性混合蒙皮 |

### `optimizer/rig.py`

```python
@dataclass
class Rig:
    joints:  np.ndarray   # (J, 3) 静置姿态关节位置
    weights: np.ndarray   # (V, J) 蒙皮权重
    parents: list         # 长度 J；根节点为 None，其余为 int 索引
    names:   np.ndarray | None  # (J,) 关节名称，可选

def load_rig(npz_path) -> Rig
```

### `optimizer/fit.py`

```python
def fit_sequence(
    rig:       Rig,
    verts_seq: np.ndarray,   # (T, V, 3)
    v0:        np.ndarray,   # (V, 3) rest pose 顶点坐标
    n_iters:   int   = 500,
    lr:        float = 1e-2,
    device:    str   = 'cuda',
    verbose:   bool  = True,
) -> np.ndarray              # (T, V, 3) float32
```

- 若第 0 帧与传入的 `v0` 相同，直接复制跳过拟合；否则也参与梯度下降拟合。
- 所有需要拟合的帧均以上一帧的 `axis_angle` / `root_trans` 作为热启动初值（第一个拟合帧初始化为零）。
- 损失函数：预测顶点与目标顶点之间的均方 L2 距离。
- 优化器：Adam。
- 返回 `n_iters` 次迭代中损失最低时对应的顶点结果（最优损失追踪）。

### `run.py`（单条序列）

给定 rig、rest pose 顶点（v0）和 mesh sequence，输出 fitted sequence。

```
python run.py --rig rig.npz --verts verts.npy --v0 v0.npy --output fitted.npy
```

参数：`--rig`、`--verts`、`--v0`、`--output`、`--n_iters`、`--lr`、`--device`。

### `batch.py`（批量处理）

负责从 HDF5 加载数据、管理目录、采样序列，批量调用 `fit_sequence`。

```
# Training Reconstruction
python batch.py --mode reconstruction \
  --rig_dir rigs --hdf5 dt4d.hdf5 --output_dir results/output

# Cross-Motion Transfer（随机采样同动物的另一条序列）
python batch.py --mode transfer \
  --rig_dir rigs --hdf5 dt4d.hdf5 --output_dir results/output
```

参数：`--mode reconstruction / transfer`、`--rig_dir`、`--hdf5`、`--output_dir`、`--n_iters`、`--lr`、`--device`、`--seed`。已处理的序列自动跳过。

---

## 拟合算法

对每一帧 t = 1 … T-1：

1. **热启动**：用上一帧的解初始化 `axis_angle` 和 `root_trans`（t=1 时初始化为零）。
2. **运行 `n_iters` 步 Adam**：
   - `R = axis_angle_to_matrix(axis_angle)`                          → `(J, 3, 3)`
   - `T_mats = forward_kinematics(joints, parents, R, root_trans)`   → `(J, 4, 4)`
   - `v_pred = lbs(v0, W, T_mats)`                                   → `(V, 3)`
   - `loss = mean((v_pred - target)²)`
   - `loss.backward(); optimizer.step()`
3. **记录 `best_verts`**（迄今为止损失最低时的预测值）。
4. 将 `best_verts` 存入输出数组的第 t 个位置。

---

## Batch 模式

### 两种模式

rig 目录结构与 HDF5 完全对应，无需字符串解析：

```
rigs/
├── bear84Q/
│   └── bear84Q_Landing/
│       └── rig.npz
└── deerOMG/
    └── deerOMG_death2/
        └── rig.npz
```

**Training Reconstruction**（同一序列重建）
- 每个 `rig.npz` fit 它自己对应的序列
- `rigs/bear84Q/bear84Q_Landing/rig.npz` → fit HDF5 中的 `bear84Q/bear84Q_Landing`

**Cross-Motion Transfer**（跨动作迁移）
- 每个 `rig.npz` 随机采样同一只动物的另一条序列进行 fit
- `rigs/bear84Q/bear84Q_Landing/rig.npz` → 随机选 `bear84Q/bear84Q_AttackrunRM`

### 输出目录结构

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

`meta.json` 记录本次 transfer 使用的 rig 来源和目标序列：
```json
{"rig": "bear84Q_Landing", "target": "bear84Q_AttackrunRM"}
```

### CLI 用法

```
# Training Reconstruction（同一序列重建）
python batch.py --mode reconstruction \
  --rig_dir rigs --hdf5 dt4d.hdf5 --output_dir results/output

# Cross-Motion Transfer（跨动作迁移）
python batch.py --mode transfer \
  --rig_dir rigs --hdf5 dt4d.hdf5 --output_dir results/output
```

参数：`--mode reconstruction / transfer`、`--rig_dir`、`--hdf5`、`--output_dir`、`--n_iters`、`--lr`、`--device`、`--seed`。已处理的序列自动跳过。

---

## 依赖项

- `torch`（≥ 2.0）— 可微分 FK/LBS 及 Adam 优化器
- `numpy`
- `h5py` — 从 `dt4d.hdf5` 加载网格序列
- `tqdm` — 进度条显示
