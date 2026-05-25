#!/usr/bin/env python
"""
generate_rot6d.py
-----------------
Run inference and conversion in one step, outputting fixed 125-frame rot6d-format npy files.

Output files:
  <output_dir>/<motion_name>/results.npy        — raw xyz inference result (intermediate file)
  <output_dir>/<motion_name>/results_rot6d.npy  — rot6d format (final output)
    motion.shape = (N, 23, 6, 125)
    layout: [0:22] = SMPL first 22 joints in rot6d format, [22] = root translation

Usage：
  python generate_rot6d.py \\
    --model_path ./save/1012_test/model000875235.pt \\
    --prompt "a person walks forward" \\
    --num_samples 1 \\
    --output_dir ./output \\
    [--device 0] \\
    [--keep_xyz]

"""

import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.fixseed import fixseed
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders.tensors import collate

N_FRAMES = 125


def parse_extra_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--prompt',        type=str, required=True)
    p.add_argument('--detailed_text', type=str, default='')
    p.add_argument('--num_samples',   type=int, default=1)
    p.add_argument('--keep_xyz',      action='store_true',
                   help='Keep intermediate xyz results.npy; by default it is removed after inference.')
    known, remaining = p.parse_known_args()
    sys.argv[1:] = remaining
    return known


# ──────────────────────────────────────────────
# ──────────────────────────────────────────────
def run_inference(args, prompt_args, out_dir):
    n_frames   = N_FRAMES
    max_frames = 196 if args.dataset in ['kit', 'humanml', 'detailed_text'] else 60

    texts = [prompt_args.prompt] * prompt_args.num_samples

    if prompt_args.detailed_text.strip():
        detailed_texts = [prompt_args.detailed_text] * prompt_args.num_samples
    else:
        default_dt = ('<SEP> <Motionless> <SEP> <SEP> <Motionless> <SEP> '
                      'Raise your right arm over your head and kick your left leg. '
                      '<SEP> <Motionless> <SEP> <Motionless> <SEP>')
        print("[INFO] detailed_text not provided, using default placeholder.")
        detailed_texts = [default_dt] * prompt_args.num_samples

    collate_args = [{'inp': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames}
                    ] * prompt_args.num_samples
    collate_args = [dict(a, text=t) for a, t in zip(collate_args, texts)]
    collate_args = [dict(a, detailed_text=dt) for a, dt in zip(collate_args, detailed_texts)]
    _, model_kwargs = collate(collate_args)

    dist_util.setup_dist(args.device)

    print("[INFO] Loading dataset metadata...")
    data = get_dataset_loader(name=args.dataset, batch_size=prompt_args.num_samples,
                              num_frames=max_frames)

    print("[INFO] Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    print(f"[INFO] Loading checkpoint: {args.model_path}")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)
    model.to(dist_util.dev())
    model.eval()

    for k, v in model_kwargs['y'].items():
        if torch.is_tensor(v):
            model_kwargs['y'][k] = v.to(dist_util.dev())

    all_joints, all_lengths = [], []

    for rep_i in range(args.num_repetitions):
        print(f"\n[INFO] Sampling rep {rep_i+1}/{args.num_repetitions} ...")

        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = (
                torch.ones(prompt_args.num_samples, device=dist_util.dev()) * args.guidance_param
            )

        sample = diffusion.p_sample_loop(
            model,
            (prompt_args.num_samples, model.njoints, model.nfeats, n_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )

        if model.data_rep == 'hml_vec':
            n_joints = 22 if sample.shape[1] == 263 else 21
            sample = data.dataset.t2m_dataset.inv_transform(
                sample.cpu().permute(0, 2, 3, 1)).float()
            sample = recover_from_ric(sample, n_joints)
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)
            joints_xyz = sample.cpu().numpy()   # (B, n_joints, 3, T)
        else:
            rot2xyz_mask = (
                None if model.data_rep == 'xyz'
                else model_kwargs['y']['mask'].reshape(prompt_args.num_samples, n_frames).bool()
            )
            xyz = model.rot2xyz(
                x=sample, mask=rot2xyz_mask, pose_rep=model.data_rep,
                glob=True, translation=True, jointstype='smpl',
                vertstrans=True, betas=None, beta=0, glob_rot=None,
                get_rotations_back=False,
            )
            joints_xyz = xyz.cpu().numpy()

        all_joints.append(joints_xyz)
        all_lengths.append(np.array([n_frames] * prompt_args.num_samples))

    all_joints  = np.concatenate(all_joints,  axis=0)  # (N, n_joints, 3, T)
    all_lengths = np.concatenate(all_lengths, axis=0)

    npy_path = os.path.join(out_dir, 'results.npy')
    np.save(npy_path, {
        'motion':          all_joints,
        'text':            texts,
        'detailed_text':   detailed_texts,
        'lengths':         all_lengths,
        'num_samples':     prompt_args.num_samples,
        'num_repetitions': args.num_repetitions,
    })
    print(f"[INFO] Saved xyz results.npy -> {npy_path}  shape={all_joints.shape}")
    return npy_path


# ──────────────────────────────────────────────
# ──────────────────────────────────────────────
def run_convert(npy_path, out_dir, device_id, use_cuda):
    from visualize.simplify_loc2rot import joints2smpl

    data   = np.load(npy_path, allow_pickle=True).item()
    motion = data['motion']               # (N, njoints, 3, T)
    N, njoints, nfeats, T = motion.shape
    print(f"\n[INFO] Converting xyz→rot6d: motion.shape={motion.shape}")

    all_rot6d, all_lengths = [], []

    for i in range(N):
        num_frames = int(data['lengths'][i])
        print(f"  [{i+1}/{N}] Running SMPLify ...")

        j2s = joints2smpl(num_frames=T, device_id=device_id, cuda=use_cuda)
        motion_tensor, _ = j2s.joint2smpl(motion[i].transpose(2, 0, 1))
        # motion_tensor: (1, 25, 6, T)
        rot6d_full = motion_tensor.cpu().numpy()[0]   # (25, 6, T)
        rot6d_22 = np.concatenate([
            rot6d_full[:22],
            rot6d_full[24:25],   # (1,  6, T) root translation
        ], axis=0)               # (23, 6, T)
        all_rot6d.append(rot6d_22)
        all_lengths.append(num_frames)

    rot6d_stack = np.stack(all_rot6d, axis=0)   # (N, 23, 6, T)
    print(f"[INFO] rot6d motion.shape = {rot6d_stack.shape}")

    out_data = dict(data)
    out_data['motion']  = rot6d_stack
    out_data['lengths'] = np.array(all_lengths)

    rot6d_path = os.path.join(out_dir, 'results_rot6d.npy')
    np.save(rot6d_path, out_data)
    print(f"[INFO] Saved results_rot6d.npy -> {rot6d_path}")
    print(f"       motion.shape = {rot6d_stack.shape}  (N, 23, 6, T)")
    print(f"       layout: [0:22]=SMPL first 22 joints in rot6d format, [22]=root translation")
    return rot6d_path


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────
def main():
    prompt_args = parse_extra_args()

    args = generate_args()
    args.num_samples    = prompt_args.num_samples
    args.batch_size     = prompt_args.num_samples

    fixseed(args.seed)

    use_cuda  = torch.cuda.is_available()
    device_id = int(args.device) if hasattr(args, 'device') and args.device is not None else 0

    motion_name = (
        "".join(x for x in prompt_args.prompt if x.isalnum() or x in " _-")
        .strip().replace(" ", "_")
    )[:64] or "motion_output"

    out_dir = os.path.join(args.output_dir, motion_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Output dir: {os.path.abspath(out_dir)}")

    npy_path = run_inference(args, prompt_args, out_dir)

    rot6d_path = run_convert(npy_path, out_dir, device_id, use_cuda)

    if not prompt_args.keep_xyz:
        os.remove(npy_path)
        print(f"[INFO] Removed intermediate results.npy")

    print(f"\n[SUCCESS] Final output: {rot6d_path}")


if __name__ == '__main__':
    main()
