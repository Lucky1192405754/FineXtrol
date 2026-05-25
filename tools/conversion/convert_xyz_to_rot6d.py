"""
Convert xyz-format results.npy (nfeats=3) to rot6d format (nfeats=6),
and save it as a new npy file for the downstream renderer.

Output motion.shape = (N, 23, 6, T)
  layout:
    [0:22]  = SMPL first 22 joints in rot6d format (HumanML3D joint order)
    [22]    = root translation (first 3 dimensions are xyz; last 3 dimensions are 0)

Usage：
    python convert_xyz_to_rot6d.py \\
        --input  ./output/Faces_.../results.npy \\
        --output ./output/Faces_.../results_rot6d.npy \\
        --device 0
"""

import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True, help='Input xyz results.npy path')
    parser.add_argument('--output', required=True, help='Output rot6d npy path')
    parser.add_argument('--device', type=int, default=0, help='GPU id，-1 means CPU')
    parser.add_argument('--cuda',   action='store_true', default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    use_cuda = args.cuda and torch.cuda.is_available()
    device_id = args.device if use_cuda else -1

    print(f"Loading: {args.input}")
    data = np.load(args.input, allow_pickle=True).item()

    motion = data['motion']           # (N, njoints, nfeats, T)
    N, njoints, nfeats, T = motion.shape
    print(f"  motion.shape = {motion.shape}  (N={N}, njoints={njoints}, nfeats={nfeats}, T={T})")

    if nfeats == 6:
        print("  nfeats=6, already rot6d. No conversion needed.")
        print(f"  Saving as-is to {args.output}")
        np.save(args.output, data)
        return

    if nfeats != 3:
        print(f"  [ERROR] Unsupported nfeats={nfeats}, expected 3 or 6.")
        sys.exit(1)

    from visualize.simplify_loc2rot import joints2smpl

    all_rot6d = []
    all_lengths = []

    for i in range(N):
        print(f"\n[{i+1}/{N}] Running SMPLify for sample {i} ...")
        num_frames = int(data['lengths'][i])

        j2s = joints2smpl(num_frames=T, device_id=device_id, cuda=use_cuda)
        # input: (T, njoints, 3)
        motion_tensor, _ = j2s.joint2smpl(motion[i].transpose(2, 0, 1))
        # motion_tensor: (1, 25, 6, T)
        rot6d_full = motion_tensor.cpu().numpy()[0]   # (25, 6, T)
        rot6d_22 = np.concatenate([
            rot6d_full[:22],
            rot6d_full[24:25],    # (1,  6, T) — root translation
        ], axis=0)                # (23, 6, T)
        all_rot6d.append(rot6d_22)
        all_lengths.append(num_frames)

    # stack -> (N, 23, 6, T)
    rot6d_stack = np.stack(all_rot6d, axis=0)
    print(f"\nConverted motion.shape = {rot6d_stack.shape}")

    out_data = dict(data)
    out_data['motion'] = rot6d_stack
    out_data['lengths'] = np.array(all_lengths)
    out_data['num_samples'] = N

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.save(args.output, out_data)
    print(f"\nSaved rot6d npy to: {args.output}")
    print(f"  motion.shape = {rot6d_stack.shape}  (N, 23, 6, T)")
    print(f"  layout: [0:22] = SMPL first 22 joints in rot6d format (HumanML3D order), [22] = root translation")


if __name__ == '__main__':
    main()
