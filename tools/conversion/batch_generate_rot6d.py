#!/usr/bin/env python
"""
batch_generate_rot6d.py
-----------------------
Batch inference script: read prompts from generate_rot6d_prompt.json,
Run each prompt 3 times and output rot6d npy files.

Output structure:
  ./output_yuhang/<key>/sample_00_rot6d.npy
  ./output_yuhang/<key>/sample_01_rot6d.npy
  ./output_yuhang/<key>/sample_02_rot6d.npy

Each npy file motion.shape = (1, 23, 6, 125)

Usage：
  python batch_generate_rot6d.py \
    --model_path ./save/1012_test/model000875235.pt \
    --json_path  ./generate_rot6d_prompt.json \
    --output_dir ./output_yuhang \
    --runs_per_prompt 3 \
    --device 0
"""

import os
import sys
import json
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

DEFAULT_DETAILED_TEXT = (
    '<SEP> <Motionless> <SEP> <SEP> <Motionless> <SEP> '
    'Raise your right arm over your head and kick your left leg. '
    '<SEP> <Motionless> <SEP> <Motionless> <SEP>'
)


def parse_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--json_path',        type=str,  default='./generate_rot6d_prompt.json')
    p.add_argument('--output_dir',       type=str,  default='./output_yuhang')
    p.add_argument('--runs_per_prompt',  type=int,  default=3)
    p.add_argument('--device',           type=int,  default=0)
    p.add_argument('--resume',           action='store_true',
                   help='Skip already generated samples and resume from checkpoints.')
    known, remaining = p.parse_known_args()
    sys.argv[1:] = remaining
    return known


# ──────────────────────────────────────────────
# ──────────────────────────────────────────────
def load_model(args, data, device):
    model, diffusion = create_model_and_diffusion(args, data)
    print(f"[INFO] Loading checkpoint: {args.model_path}")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)
    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)
    model.to(device)
    model.eval()
    return model, diffusion


# ──────────────────────────────────────────────
# ──────────────────────────────────────────────
def run_one_sample(model, diffusion, data, args, prompt, device):
    detailed_text = DEFAULT_DETAILED_TEXT
    collate_args = [{'inp': torch.zeros(N_FRAMES), 'tokens': None, 'lengths': N_FRAMES,
                     'text': prompt, 'detailed_text': detailed_text}]
    _, model_kwargs = collate(collate_args)

    for k, v in model_kwargs['y'].items():
        if torch.is_tensor(v):
            model_kwargs['y'][k] = v.to(device)

    if args.guidance_param != 1:
        model_kwargs['y']['scale'] = (
            torch.ones(1, device=device) * args.guidance_param
        )

    with torch.no_grad():
        sample = diffusion.p_sample_loop(
            model,
            (1, model.njoints, model.nfeats, N_FRAMES),
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
        joints_xyz = sample.cpu().numpy()   # (1, n_joints, 3, T)
    else:
        rot2xyz_mask = (
            None if model.data_rep == 'xyz'
            else model_kwargs['y']['mask'].reshape(1, N_FRAMES).bool()
        )
        xyz = model.rot2xyz(
            x=sample, mask=rot2xyz_mask, pose_rep=model.data_rep,
            glob=True, translation=True, jointstype='smpl',
            vertstrans=True, betas=None, beta=0, glob_rot=None,
            get_rotations_back=False,
        )
        joints_xyz = xyz.cpu().numpy()

    return joints_xyz  # (1, n_joints, 3, T)


# ──────────────────────────────────────────────
# ──────────────────────────────────────────────
def convert_to_rot6d(joints_xyz, device_id, use_cuda):
    """joints_xyz: (1, n_joints, 3, T) → returns rot6d: (1, 23, 6, T)"""
    from visualize.simplify_loc2rot import joints2smpl

    T = joints_xyz.shape[-1]
    j2s = joints2smpl(num_frames=T, device_id=device_id, cuda=use_cuda)
    motion_tensor, _ = j2s.joint2smpl(joints_xyz[0].transpose(2, 0, 1))  # (T, n_joints, 3)
    # motion_tensor: (1, 25, 6, T)
    rot6d_full = motion_tensor.cpu().numpy()[0]   # (25, 6, T)
    rot6d_22 = np.concatenate([
        rot6d_full[:22],
        rot6d_full[24:25],   # (1,  6, T) root translation
    ], axis=0)               # (23, 6, T)
    return rot6d_22[np.newaxis]  # (1, 23, 6, T)


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────
def main():
    extra = parse_args()

    args = generate_args()
    args.batch_size = 1
    args.num_samples = 1

    fixseed(args.seed)

    use_cuda  = torch.cuda.is_available()
    device_id = extra.device
    device    = torch.device(f'cuda:{device_id}' if use_cuda else 'cpu')
    dist_util.setup_dist(args.device)

    with open(extra.json_path, 'r') as f:
        prompt_dict = json.load(f)

    keys = list(prompt_dict.keys())
    print(f"[INFO] Total prompts: {len(keys)}, runs per prompt: {extra.runs_per_prompt}")
    print(f"[INFO] Total samples to generate: {len(keys) * extra.runs_per_prompt}")

    print("[INFO] Loading dataset metadata...")
    max_frames = 196 if args.dataset in ['kit', 'humanml', 'detailed_text'] else 60
    data = get_dataset_loader(name=args.dataset, batch_size=1, num_frames=max_frames)

    model, diffusion = load_model(args, data, device)

    total     = len(keys) * extra.runs_per_prompt
    finished  = 0
    skipped   = 0

    for ki, key in enumerate(keys):
        prompt = prompt_dict[key]
        if isinstance(prompt, list):
            prompt = prompt[0]

        key_dir = os.path.join(extra.output_dir, key)
        os.makedirs(key_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"[{ki+1}/{len(keys)}] key: {key}")
        print(f"  prompt: {prompt[:80]}...")

        for run_i in range(extra.runs_per_prompt):
            out_path = os.path.join(key_dir, f'sample_{run_i:02d}_rot6d.npy')

            if extra.resume and os.path.exists(out_path):
                print(f"  [SKIP] run {run_i:02d} already exists: {out_path}")
                skipped += 1
                finished += 1
                continue

            print(f"  [RUN {run_i:02d}] Inferencing ...")
            joints_xyz = run_one_sample(model, diffusion, data, args, prompt, device)

            print(f"  [RUN {run_i:02d}] Converting xyz→rot6d ...")
            rot6d = convert_to_rot6d(joints_xyz, device_id, use_cuda)

            out_data = {
                'motion':  rot6d,           # (1, 23, 6, 125)
                'text':    [prompt],
                'lengths': np.array([N_FRAMES]),
                'key':     key,
                'run_idx': run_i,
            }
            np.save(out_path, out_data)
            finished += 1
            print(f"  [DONE] {out_path}  motion.shape={rot6d.shape}  ({finished}/{total})")

    print(f"\n{'='*60}")
    print(f"[SUCCESS] All done. {finished} samples generated ({skipped} skipped).")
    print(f"Output dir: {os.path.abspath(extra.output_dir)}")


if __name__ == '__main__':
    main()
