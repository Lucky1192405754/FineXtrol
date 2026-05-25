"""
motionlcm_save_joints.py
------------------------
Run inference only, without rendering.

Output an .npz file containing:
  joints      : (N_samples, T, n_joints, 3)  —— xyz joint coordinates
  text        : (N_samples,)                 —— text prompts
  lengths     : (N_samples,)                 —— valid frame counts

Usage example：
  python motionlcm_save_joints.py \\
    --prompt "a person walks forward" \\
    --output_dir ./output \\
    --model_path ./save/my_model/model.pt \\
    [Other generate_args parameters are the same as the original version.]
"""

import os
import shutil
import argparse
import numpy as np
import torch
import sys

from utils.fixseed import fixseed
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders.tensors import collate


def main():
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument("--prompt",        type=str, required=True,
                             help="Coarse text description")
    temp_parser.add_argument("--detailed_text", type=str, default='',
                             help="Fine-grained text description (optional)")
    temp_parser.add_argument("--num_samples",   type=int, default=1,
                             help="Number of samples generated per inference run")
    prompt_args, remaining_argv = temp_parser.parse_known_args()
    sys.argv[1:] = remaining_argv

    args = generate_args()
    args.prompt        = prompt_args.prompt
    args.detailed_text = prompt_args.detailed_text
    args.num_samples   = prompt_args.num_samples

    fixseed(args.seed)
    args.batch_size = args.num_samples

    max_frames = 196 if args.dataset in ['kit', 'humanml', 'detailed_text'] else 60
    n_frames   = 196
    print(f"[INFO] Motion length fixed to {n_frames} frames.")

    texts = [args.prompt] * args.num_samples

    if args.detailed_text.strip():
        detailed_texts = [args.detailed_text] * args.num_samples
    else:
        default_dt = ('<SEP> <Motionless> <SEP> <SEP> <Motionless> <SEP> '
                      'Raise your right arm over your head and kick your left leg. '
                      '<SEP> <Motionless> <SEP> <Motionless> <SEP>')
        print("[INFO] detailed_text not provided, using default placeholder.")
        detailed_texts = [default_dt] * args.num_samples

    collate_args = [{'inp': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames}
                    ] * args.num_samples
    collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
    collate_args = [dict(arg, detailed_text=dt)  for arg, dt  in zip(collate_args, detailed_texts)]
    _, model_kwargs = collate(collate_args)

    dist_util.setup_dist(args.device)

    print("[INFO] Loading dataset metadata...")
    data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size,
                              num_frames=max_frames)

    print("[INFO] Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    print(f"[INFO] Loading checkpoint from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)
    model.to(dist_util.dev())
    model.eval()

    for k, v in model_kwargs['y'].items():
        if torch.is_tensor(v):
            model_kwargs['y'][k] = v.to(dist_util.dev())

    all_joints  = []   # list of (B, n_joints, 3, T) numpy
    all_text    = []
    all_detail  = []
    all_lengths = []

    for rep_i in range(args.num_repetitions):
        print(f"\n[INFO] Sampling repetition {rep_i + 1}/{args.num_repetitions} ...")

        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = (
                torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
            )

        sample = diffusion.p_sample_loop(
            model,
            (args.batch_size, model.njoints, model.nfeats, n_frames),
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
                sample.cpu().permute(0, 2, 3, 1)
            ).float()
            sample = recover_from_ric(sample, n_joints)
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)
            # sample: (B, n_joints, 3, T)
            joints_xyz = sample.cpu().numpy()   # (B, n_joints, 3, T)

        else:
            rot2xyz_pose_rep = 'xyz' if model.data_rep == 'xyz' else model.data_rep
            rot2xyz_mask = (
                None if rot2xyz_pose_rep == 'xyz'
                else model_kwargs['y']['mask'].reshape(args.batch_size, n_frames).bool()
            )
            xyz = model.rot2xyz(
                x=sample, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep,
                glob=True, translation=True, jointstype='smpl',
                vertstrans=True, betas=None, beta=0, glob_rot=None,
                get_rotations_back=False,
            )
            joints_xyz = xyz.cpu().numpy()  # (B, n_joints, 3, T)

        all_joints.append(joints_xyz)
        all_lengths.append(np.array([n_frames] * args.batch_size))
        all_text.extend(texts)
        all_detail.extend(detailed_texts)

    all_joints  = np.concatenate(all_joints,  axis=0)  # (N_total, n_joints, 3, T)
    all_lengths = np.concatenate(all_lengths, axis=0)  # (N_total,)

    all_joints_TNJ3 = all_joints.transpose(0, 3, 1, 2)  # (N, T, n_joints, 3)

    print(f"\n[INFO] all_joints shape (N, T, n_joints, 3) = {all_joints_TNJ3.shape}")
    print(f"[INFO]   N_total  = {all_joints_TNJ3.shape[0]}")
    print(f"[INFO]   T_frames = {all_joints_TNJ3.shape[1]}")
    print(f"[INFO]   n_joints = {all_joints_TNJ3.shape[2]}")
    print(f"[INFO]   3 (xyz)  = {all_joints_TNJ3.shape[3]}")

    os.makedirs(args.output_dir, exist_ok=True)
    motion_name = (
        "".join(x for x in args.prompt if x.isalnum() or x in " _-")
        .strip().replace(" ", "_")
    )[:64] or "motion_output"

    out_dir = os.path.join(args.output_dir, motion_name)
    os.makedirs(out_dir, exist_ok=True)

    npy_path = os.path.join(out_dir, "results.npy")
    np.save(npy_path, {
        'motion':          all_joints,
        'text':            all_text,
        'detailed_text':   all_detail,
        'lengths':         all_lengths,
        'num_samples':     args.num_samples,
        'num_repetitions': args.num_repetitions,
    })
    print(f"[INFO] Saved results.npy to [{npy_path}]")

    npz_path = os.path.join(out_dir, "joints.npz")
    np.savez(npz_path,
             joints=all_joints_TNJ3,    # (N, T, n_joints, 3)
             text=np.array(all_text),
             lengths=all_lengths)
    print(f"[INFO] Saved joints.npz to [{npz_path}]")
    print(f"\n[SUCCESS] Done. All outputs under: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
