"""
bvh_to_rot6d.py
---------------
Convert BVH files output by MoMask to rot6d-format npy files.

Pipeline：
  BVH (ZYX Euler) → FK → 22joints xyzworld coordinates (T,22,3)
                        → joints2smpl (SMPLify)
                        → rot6d (T, 25, 6)
                        → keep the first 22joints + root_translation
                        → save npy  motion.shape = (1, 23, 6, T)

Output layout, fully consistent with OmniControl/MDM batch generation results:
  [0:22]  SMPL first 22 joints in rot6d format (HumanML3D order)
  [22]    root translation（first 3 dimensions xyz，last 3 dimensions 0）

Usage：
  python bvh_to_rot6d.py --input sample.bvh --output momask_sample_rot6d.npy --device 0
"""

import os, sys, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ──────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────

def parse_bvh(filepath):
    """
    Parse a BVH file and return::
        joint_names  : list[str]，in tree depth-first order
        offsets      : dict{name: np.array(3,)}, T-pose bone offsets in meters
        parents      : dict{name: str or None}
        channel_order: list[(joint_name, ['Xposition',...] or ['Zrotation',...])]
        frames_data  : np.array(T, total_channels)
        frame_time   : float
    """
    with open(filepath, 'r') as f:
        lines = [l.strip() for l in f.readlines()]

    joint_names  = []
    offsets      = {}
    parents      = {}
    channel_order = []   # list of (joint_name, [ch1, ch2, ...])
    stack        = []

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('ROOT') or line.startswith('JOINT'):
            parts = line.split()
            jname = parts[1]
            joint_names.append(jname)
            parents[jname] = stack[-1] if stack else None
        elif line.startswith('End Site'):
            i += 1  # {
            i += 1  # OFFSET
            i += 1  # }
            i += 1
            continue
        elif line.startswith('{'):
            if joint_names:
                stack.append(joint_names[-1])
        elif line.startswith('}'):
            if stack:
                stack.pop()
        elif line.startswith('OFFSET'):
            vals = list(map(float, line.split()[1:]))
            jname = joint_names[-1] if joint_names else None
            if jname:
                offsets[jname] = np.array(vals)
        elif line.startswith('CHANNELS'):
            parts = line.split()
            n_ch  = int(parts[1])
            chs   = parts[2:2+n_ch]
            jname = joint_names[-1]
            channel_order.append((jname, chs))
        elif line.startswith('MOTION'):
            i += 1
            n_frames   = int(lines[i].split(':')[1].strip())
            i += 1
            frame_time = float(lines[i].split(':')[1].strip())
            i += 1
            frames_data = []
            for _ in range(n_frames):
                frames_data.append(list(map(float, lines[i].split())))
                i += 1
            frames_data = np.array(frames_data, dtype=np.float32)
            break
        i += 1

    return joint_names, offsets, parents, channel_order, frames_data, frame_time


# ──────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────

def euler_zyx_to_rotmat(z_deg, y_deg, x_deg):
    """ZYX Euler angles in degrees to a 3x3 rotation matrix"""
    z = np.deg2rad(z_deg)
    y = np.deg2rad(y_deg)
    x = np.deg2rad(x_deg)

    Rz = np.array([[np.cos(z), -np.sin(z), 0],
                   [np.sin(z),  np.cos(z), 0],
                   [0,          0,         1]], dtype=np.float32)
    Ry = np.array([[ np.cos(y), 0, np.sin(y)],
                   [ 0,         1, 0        ],
                   [-np.sin(y), 0, np.cos(y)]], dtype=np.float32)
    Rx = np.array([[1, 0,          0         ],
                   [0, np.cos(x), -np.sin(x) ],
                   [0, np.sin(x),  np.cos(x) ]], dtype=np.float32)
    return Rz @ Ry @ Rx


def forward_kinematics(joint_names, offsets, parents, channel_order, frames_data):
    """
    FK: compute world-coordinate positions for every joint in every frame.
    Return positions: np.array(T, N_joints, 3)
    """
    T      = frames_data.shape[0]
    N      = len(joint_names)
    jidx   = {n: i for i, n in enumerate(joint_names)}

    positions = np.zeros((T, N, 3), dtype=np.float32)
    rotmats   = np.zeros((T, N, 3, 3), dtype=np.float32)

    for t in range(T):
        frame   = frames_data[t]
        col_ptr = 0
        world_R = {}   # joint_name -> 3×3
        world_t = {}   # joint_name -> (3,)

        for jname, chs in channel_order:
            vals = {c: frame[col_ptr + k] for k, c in enumerate(chs)}
            col_ptr += len(chs)

            tx = vals.get('Xposition', 0.0)
            ty = vals.get('Yposition', 0.0)
            tz = vals.get('Zposition', 0.0)
            local_t = np.array([tx, ty, tz], dtype=np.float32)

            rz = vals.get('Zrotation', 0.0)
            ry = vals.get('Yrotation', 0.0)
            rx = vals.get('Xrotation', 0.0)
            local_R = euler_zyx_to_rotmat(rz, ry, rx)

            parent = parents[jname]
            if parent is None:
                world_R[jname] = local_R
                world_t[jname] = local_t
            else:
                pR = world_R[parent]
                pt = world_t[parent]
                world_R[jname] = pR @ local_R
                world_t[jname] = pt + pR @ offsets[jname]

            ji = jidx[jname]
            positions[t, ji] = world_t[jname]
            rotmats[t, ji]   = world_R[jname]

    return positions   # (T, N_joints, 3)


# ──────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────

SMPL_22_NAMES = [
    'pelvis',        # 0   → Hips
    'left_hip',      # 1   → LeftUpLeg
    'right_hip',     # 2   → RightUpLeg
    'spine1',        # 3   → Spine
    'left_knee',     # 4   → LeftLeg
    'right_knee',    # 5   → RightLeg
    'spine2',        # 6   → Spine1
    'left_ankle',    # 7   → LeftFoot
    'right_ankle',   # 8   → RightFoot
    'spine3',        # 9   → Spine2
    'left_foot',     # 10  → LeftToe
    'right_foot',    # 11  → RightToe
    'neck',          # 12  → Neck
    'left_collar',   # 13  → LeftShoulder
    'right_collar',  # 14  → RightShoulder
    'head',          # 15  → Head
    'left_shoulder', # 16  → LeftArm
    'right_shoulder',# 17  → RightArm
    'left_elbow',    # 18  → LeftForeArm
    'right_elbow',   # 19  → RightForeArm
    'left_wrist',    # 20  → LeftHand
    'right_wrist',   # 21  → RightHand
]

BVH_TO_SMPL = [
    'Hips',
    'LeftUpLeg',
    'RightUpLeg',
    'Spine',
    'LeftLeg',
    'RightLeg',
    'Spine1',
    'LeftFoot',
    'RightFoot',
    'Spine2',
    'LeftToe',
    'RightToe',
    'Neck',
    'LeftShoulder',
    'RightShoulder',
    'Head',
    'LeftArm',
    'RightArm',
    'LeftForeArm',
    'RightForeArm',
    'LeftHand',
    'RightHand',
]


def reorder_to_smpl22(positions, joint_names):
    """
    positions: (T, N_bvh, 3)
    joint_names: BVH joint-name list
    Return: (T, 22, 3), ordered by the SMPL 22-joint layout
    """
    jidx = {n: i for i, n in enumerate(joint_names)}
    missing = [n for n in BVH_TO_SMPL if n not in jidx]
    if missing:
        raise ValueError(f"Missing joints in BVH file: {missing}")
    idx = [jidx[n] for n in BVH_TO_SMPL]
    return positions[:, idx, :]   # (T, 22, 3)


# ──────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────

def xyz_to_rot6d(joints_xyz_T22, device_id, use_cuda):
    """
    joints_xyz_T22: np.array(T, 22, 3)
    Return: np.array(1, 23, 6, T)
      [0:22] SMPL first 22 joints in rot6d format
      [22]   root translation (first 3 dimensionsxyz, last 3 dimensions0)
    """
    from visualize.simplify_loc2rot import joints2smpl

    T = joints_xyz_T22.shape[0]
    j2s = joints2smpl(num_frames=T, device_id=device_id, cuda=use_cuda)
    motion_tensor, _ = j2s.joint2smpl(joints_xyz_T22)   # (1, 25, 6, T)

    rot6d_full = motion_tensor.cpu().numpy()[0]           # (25, 6, T)
    rot6d_23 = np.concatenate([
        rot6d_full[:22],
        rot6d_full[24:25],    # (1,  6, T) root translation
    ], axis=0)                # (23, 6, T)
    return rot6d_23[np.newaxis]   # (1, 23, 6, T)


# ──────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input',  required=True,  help='Input BVH file path')
    p.add_argument('--output', required=True,  help='Output rot6d npy path')
    p.add_argument('--device', type=int, default=0, help='GPU id，-1 means CPU')
    p.add_argument('--no_cuda', action='store_true', help='Force CPU usage')
    return p.parse_args()


def main():
    args   = parse_args()
    use_cuda  = (not args.no_cuda) and torch.cuda.is_available()
    device_id = args.device if use_cuda else -1

    print(f"[1/4] Parsing BVH: {args.input}")
    joint_names, offsets, parents, channel_order, frames_data, frame_time = parse_bvh(args.input)
    T = frames_data.shape[0]
    print(f"  joints: {len(joint_names)}, frames: {T}, frame_time: {frame_time:.4f}s  "
          f"({1/frame_time:.1f} fps)")

    print("[2/4] Forward kinematics → world xyz positions")
    positions = forward_kinematics(joint_names, offsets, parents, channel_order, frames_data)
    print(f"  positions.shape = {positions.shape}  (T, N_bvh, 3)")

    print("[3/4] Reordering to SMPL-22 joint order")
    joints_smpl22 = reorder_to_smpl22(positions, joint_names)
    print(f"  joints_smpl22.shape = {joints_smpl22.shape}  (T, 22, 3)")

    print("[4/4] SMPLify: xyz → rot6d")
    rot6d = xyz_to_rot6d(joints_smpl22, device_id, use_cuda)
    print(f"  rot6d.shape = {rot6d.shape}  (1, 23, 6, T)")

    out_data = {
        'motion':      rot6d,                        # (1, 23, 6, T)
        'lengths':     np.array([T]),
        'num_samples': 1,
        'source_file': os.path.abspath(args.input),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)
    np.save(args.output, out_data)
    print(f"\n[DONE] Saved to: {args.output}")
    print(f"  motion.shape = {rot6d.shape}")
    print("  layout: [0:22]=SMPL first 22 joints in rot6d format, [22]=root_translation")


if __name__ == '__main__':
    main()
