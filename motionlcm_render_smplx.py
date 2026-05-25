# motionlcm_render_smplx.py
#
#
# ------------------------------------------------------------------------------------

import os
import shutil
import argparse
import numpy as np
import torch
import sys
import pickle

import trimesh
from tqdm import tqdm

# --- SMPL-X ---
from smplx import SMPLX as _SMPLX

import utils.rotation_conversions as geometry

# --- COMMON UTILITIES ---
from utils.fixseed import fixseed
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders.tensors import collate


# =================================================================================================
# [SMPL-X WRAPPER]
# =================================================================================================

SMPLX_MODEL_PATH = "./body_models/smpl/SMPLX_NEUTRAL_2020.npz"

SMPL_BODY_JOINTS  = 23
SMPLX_BODY_JOINTS = 21
SMPLX_HAND_JOINTS = 15


class Rotation2smplx:
    """
    Convert a rot6d motion sequence into an SMPL-X vertex sequence.

    The input format is the same as Rotation2xyz:
        x: Tensor (B, njoints_with_root, nfeats=6, T)
            where the last joint channel stores root translation (first 3 dimensions are valid)
            and the preceding channels store rotations in rot6d format
        convention: translation=True, glob=True (consistent with vis_utils)

    Output:
        vertices: Tensor (B, 10475, 3, T), with the same axis order as Rotation2xyz output
    """

    def __init__(self, device='cpu'):
        self.device = device
        self._model_cache = {}  # cache by batch_size to avoid re-init

    def _get_model(self, batch_size: int):
        """Cache SMPL-X instances by batch_size because smplx 0.1.x requires batch_size at construction time."""
        if batch_size not in self._model_cache:
            model = _SMPLX(
                SMPLX_MODEL_PATH,
                use_pca=False,
                flat_hand_mean=True,
                batch_size=batch_size,
                num_betas=10,
            ).eval().to(self.device)
            self._model_cache[batch_size] = model
        return self._model_cache[batch_size]

    @property
    def faces(self):
        """Return SMPL-X face indices for building trimesh objects."""
        return self._get_model(1).faces

    def __call__(self, x: torch.Tensor, beta: float = 0.0) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (B, njoints+1, 6, T)
            The last joint is root translation using only the first 3 dimensions; the others are rot6d.

        Returns
        -------
        vertices : Tensor, shape (B, 10475, 3, T)
        """
        B, njoints_full, nfeats, T = x.shape
        assert nfeats == 6, f"Expected rot6d (nfeats=6), got {nfeats}"

        # x[:, -1, :3, :]  → root translation  (B, 3, T)
        # x[:, :-1, :, :]  → rotations rot6d   (B, njoints, 6, T)
        x_translations = x[:, -1, :3, :]          # (B, 3, T)
        x_rotations    = x[:, :-1, :, :]           # (B, njoints, 6, T)  njoints=24

        # x_rotations: (B, njoints, 6, T) → (B*T, njoints, 6)
        x_rot_bt = x_rotations.permute(0, 3, 1, 2).reshape(B * T, -1, 6)  # (B*T, njoints, 6)

        # rot6d → rotation matrix (B*T, njoints, 3, 3)
        rotmats = geometry.rotation_6d_to_matrix(x_rot_bt)  # (B*T, njoints, 3, 3)

        global_orient_mat = rotmats[:, 0, :, :]   # (B*T, 3, 3)

        body_pose_mat = rotmats[:, 1:1 + SMPLX_BODY_JOINTS, :, :]  # (B*T, 21, 3, 3)

        global_orient_aa = geometry.matrix_to_axis_angle(global_orient_mat)   # (B*T, 3)
        body_pose_aa     = geometry.matrix_to_axis_angle(body_pose_mat)        # (B*T, 21, 3)
        body_pose_aa     = body_pose_aa.reshape(B * T, SMPLX_BODY_JOINTS * 3) # (B*T, 63)

        hand_pose_zero = torch.zeros(B * T, SMPLX_HAND_JOINTS * 3,
                                     dtype=x.dtype, device=self.device)
        jaw_zero  = torch.zeros(B * T, 3, dtype=x.dtype, device=self.device)
        eye_zero  = torch.zeros(B * T, 3, dtype=x.dtype, device=self.device)
        betas     = torch.zeros(B * T, 10, dtype=x.dtype, device=self.device)
        if beta != 0.0:
            betas[:, 1] = beta

        model = self._get_model(B * T)
        with torch.no_grad():
            out = model(
                global_orient=global_orient_aa,
                body_pose=body_pose_aa,
                left_hand_pose=hand_pose_zero,
                right_hand_pose=hand_pose_zero,
                jaw_pose=jaw_zero,
                leye_pose=eye_zero,
                reye_pose=eye_zero,
                betas=betas,
            )

        vertices_bt = out.vertices  # (B*T, 10475, 3)

        vertices = vertices_bt.reshape(B, T, 10475, 3)
        vertices = vertices.permute(0, 2, 3, 1).contiguous()  # (B, 10475, 3, T)

        # x_translations: (B, 3, T) → (B, 1, 3, T)
        x_trans = x_translations - x_translations[:, :, [0]]  # zero-origin
        vertices = vertices + x_trans[:, None, :, :]           # broadcast (B,10475,3,T)

        return vertices  # (B, 10475, 3, T)


# =================================================================================================
# =================================================================================================

def step2_convert_to_smplx_sequence(npy_path, sample_i, rep_i,
                                     output_dir, motion_name,
                                     device, use_cuda, hint_data):
    """
    Use SMPL-X to convert a motion sequence into a 10475-vertex mesh sequence and save it as .pkl.

    Supports two input formats:
      - nfeats==6 (rot6d): feed directly into SMPL-X
      - nfeats==3 (xyz):   first convert to rot6d with SMPLify (joints2smpl) in vis_utils.npy2obj,
                           then feed into SMPL-X. This reuses the existing IK pipeline without extra code.

    Output: {motion_name}_smplx_mesh.pkl
          contains 'vertices': (T, 10475, 3) and 'hint': (T, 3)
    """
    print(f"\n--- Step 2 (SMPL-X): Converting skeleton data to SMPL-X Mesh PKL ---")

    try:
        motions_data = np.load(npy_path, allow_pickle=True).item()
    except Exception as e:
        print(f"  [ERROR] Cannot load {npy_path}: {e}")
        return None

    bs           = motions_data['num_samples']
    absl_idx     = rep_i * bs + sample_i
    motion_seq   = motions_data['motion'][absl_idx]   # (njoints, nfeats, T)
    real_nframes = int(motions_data['lengths'][absl_idx])

    print(f"  motion_seq.shape = {motion_seq.shape},  real_num_frames = {real_nframes}")
    njoints_full, nfeats, T = motion_seq.shape

    torch_device = torch.device(f'cuda:{device}' if use_cuda else 'cpu')
    r2smplx = Rotation2smplx(device=torch_device)

    if nfeats == 3:
        # ----------------------------------------------------------------
        # ----------------------------------------------------------------
        print(f"  nfeats=3 (xyz) detected. Running SMPLify via vis_utils.npy2obj ...")
        print(f"  (This may take a few minutes per sample)")
        from visualize import vis_utils
        try:
            npy2obj = vis_utils.npy2obj(npy_path, sample_i, rep_i,
                                        device=device, cuda=use_cuda)
        except Exception as e:
            print(f"  [ERROR] vis_utils.npy2obj failed: {e}")
            return None

        # shape: (1, njoints+1, 6, T)
        rot6d_motion = torch.tensor(npy2obj.motions['motion'], dtype=torch.float32)
        print(f"  SMPLify done. rot6d_motion.shape = {rot6d_motion.shape}")

        x = rot6d_motion.to(torch_device)   # (1, njoints+1, 6, T)

    elif nfeats == 6:
        # ----------------------------------------------------------------
        # ----------------------------------------------------------------
        print(f"  nfeats=6 (rot6d) detected. Using directly.")
        x = torch.tensor(motion_seq, dtype=torch.float32,
                         device=torch_device).unsqueeze(0)  # (1, njoints+1, 6, T)
    else:
        print(f"  [ERROR] Unsupported nfeats={nfeats}. Expected 3 (xyz) or 6 (rot6d).")
        return None

    T_actual = x.shape[-1]
    print(f"  Running Rotation2smplx forward (B=1, T={T_actual}) ...")
    try:
        vertices_all = r2smplx(x)   # (1, 10475, 3, T)
    except Exception as e:
        print(f"  [ERROR] Rotation2smplx failed: {e}")
        import traceback; traceback.print_exc()
        return None

    vertices_all = vertices_all[0]                    # (10475, 3, T)
    vertices_all = vertices_all[:, :, :real_nframes]  # (10475, 3, T_real)

    floor_y = vertices_all[:, 1, :].min().item()
    vertices_all[:, 1, :] -= floor_y

    verts_np = vertices_all.permute(2, 0, 1).cpu().numpy()  # (T, 10475, 3)

    print(f"  [DEBUG] SMPL-X vertices shape = {verts_np.shape}  (T x N_verts x 3)")
    print(f"  [DEBUG]   T (frames)          = {verts_np.shape[0]}")
    print(f"  [DEBUG]   N (vertices)        = {verts_np.shape[1]}  (SMPL-X: 10475)")
    print(f"  [DEBUG]   3 (xyz)             = {verts_np.shape[2]}")
    print(f"  [DEBUG]   dtype               = {verts_np.dtype}")
    print(f"  [DEBUG] hint_data.shape       = {np.array(hint_data).shape}")

    mesh_pkl_path = os.path.join(output_dir, f"{motion_name}_smplx_mesh.pkl")
    print(f"  Saving SMPL-X mesh pkl to [{mesh_pkl_path}]")

    try:
        data_to_save = {
            'vertices': verts_np,              # (T, 10475, 3)
            'hint':     np.array(hint_data),   # (T, 3)
            'faces':    r2smplx.faces,
            'model':    'smplx',
        }
        with open(mesh_pkl_path, 'wb') as f:
            pickle.dump(data_to_save, f)
    except Exception as e:
        print(f"  [ERROR] Failed to save pkl: {e}")
        return None

    return mesh_pkl_path


# =================================================================================================
# =================================================================================================
def step1_generate_skeleton(args, motion_name):
    """Generate skeleton motion sequences in rot6d format, same as motionlcm_render.py."""
    print("--- Step 1: Generating skeleton motion from text ---")
    fixseed(args.seed)

    args.batch_size = args.num_samples
    out_path = os.path.join(args.output_dir, motion_name)
    max_frames = 196 if args.dataset in ['kit', 'humanml', 'detailed_text'] else 60
    n_frames = 125
    print(f"Motion length is FIXED to {n_frames} frames to match model's training.")

    dist_util.setup_dist(args.device)

    print(f"Coarse-grained prompt: {args.prompt}")
    print(f"Number of samples to generate: {args.num_samples}")

    texts = [args.prompt] * args.num_samples

    if args.detailed_text.strip() != '':
        print(f"Fine-grained prompt: {args.detailed_text}")
        detailed_texts = [args.detailed_text] * args.num_samples
    else:
        default_dt = ('<SEP> <Motionless> <SEP> <SEP> <Motionless> <SEP> '
                      'Raise your right arm over your head and kick your left leg. '
                      '<SEP> <Motionless> <SEP> <Motionless> <SEP>')
        print("Fine-grained prompt not provided, using default.")
        detailed_texts = [default_dt] * args.num_samples

    collate_args = [{'inp': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames}] * args.num_samples
    collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
    collate_args = [dict(arg, detailed_text=dt) for arg, dt in zip(collate_args, detailed_texts)]

    _, model_kwargs = collate(collate_args)

    print("Loading dataset metadata...")
    data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=max_frames)

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)
    model.to(dist_util.dev())
    model.eval()

    for k, v in model_kwargs['y'].items():
        if torch.is_tensor(v):
            model_kwargs['y'][k] = v.to(dist_util.dev())

    all_motions = []
    all_lengths = []
    all_text = []
    all_detailed_text = []

    for rep_i in range(args.num_repetitions):
        print(f'### Sampling [repetitions #{rep_i+1}/{args.num_repetitions}]')
        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = (
                torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
            )

        sample_fn = diffusion.p_sample_loop
        sample = sample_fn(
            model, (args.batch_size, model.njoints, model.nfeats, n_frames),
            clip_denoised=False, model_kwargs=model_kwargs, skip_timesteps=0,
            init_image=None, progress=True, dump_steps=None, noise=None, const_noise=False,
        )

        if model.data_rep == 'hml_vec':
            n_joints = 22 if sample.shape[1] == 263 else 21
            sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
            sample = recover_from_ric(sample, n_joints)
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

        rot2xyz_pose_rep = 'xyz' if model.data_rep in ['xyz', 'hml_vec'] else model.data_rep
        rot2xyz_mask = (
            None if rot2xyz_pose_rep == 'xyz'
            else model_kwargs['y']['mask'].reshape(args.batch_size, n_frames).bool()
        )

        if rot2xyz_pose_rep == 'rot6d':
            motion_to_save = sample.clone()
        else:
            motion_to_save = model.rot2xyz(
                x=sample, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep,
                glob=True, translation=True, jointstype='smpl',
                vertstrans=True, betas=None, beta=0, glob_rot=None,
                get_rotations_back=False,
            )

        all_motions.append(motion_to_save.cpu().numpy())
        all_lengths.append(np.array([n_frames] * args.batch_size))
        all_text.extend(texts)
        all_detailed_text.extend(detailed_texts)

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    all_motions = np.concatenate(all_motions, axis=0)   # (N_total, njoints, 6, T)
    all_lengths = np.concatenate(all_lengths, axis=0)

    npy_path = os.path.join(out_path, 'results.npy')
    print(f"Saving all {len(all_motions)} generated samples to [{npy_path}]")
    np.save(npy_path, {
        'motion':           all_motions,
        'text':             all_text,
        'lengths':          all_lengths,
        'detailed_text':    all_detailed_text,
        'num_samples':      args.num_samples,
        'num_repetitions':  args.num_repetitions,
    })

    # --- DEBUG ---
    print(f"\n  [DEBUG] all_motions.shape = {all_motions.shape}")
    print(f"  [DEBUG]   rot6d model: (N_total, njoints+1, 6, T) | xyz model: (N_total, njoints, 3, T)")

    total_samples = args.num_samples * args.num_repetitions
    return npy_path, total_samples


# =================================================================================================
# [MAIN ORCHESTRATOR]
# =================================================================================================
def main():
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument("--prompt",        type=str, required=True)
    temp_parser.add_argument("--detailed_text", type=str, default='')
    temp_parser.add_argument("--num_samples",   type=int, default=10)

    prompt_args, remaining_argv = temp_parser.parse_known_args()
    sys.argv[1:] = remaining_argv

    args = generate_args()
    args.prompt        = prompt_args.prompt
    args.detailed_text = prompt_args.detailed_text
    args.num_samples   = prompt_args.num_samples

    if args.num_repetitions > 1:
        print(f"Warning: num_repetitions={args.num_repetitions}, "
              f"total motions={args.num_samples * args.num_repetitions}")

    smplx_abs = os.path.abspath(SMPLX_MODEL_PATH)
    if not os.path.exists(smplx_abs):
        print("=" * 60)
        print(f"Error: SMPL-X model file was not found.")
        print(f"Expected path: {smplx_abs}")
        print(f"Please place SMPLX_NEUTRAL_2020.npz under body_models/smpl/.")
        print("=" * 60)
        sys.exit(1)

    BLENDER_PATH = "/data/shenkeming/blender-3.6.23-linux-x64/blender"
    base_motionlcm_dir = "./Motionlcm"
    MOTIONLCM_RENDER_SCRIPT_PATH = os.path.join(base_motionlcm_dir, "render.py")
    SMPLX_FACES_PATH = "./deps/smpl_models/smplx/smplx.faces"

    if not os.path.exists(BLENDER_PATH):
        print("=" * 50)
        print(f"Error: Blender executable was not found: {BLENDER_PATH}")
        print("=" * 50)
        sys.exit(1)

    if not os.path.exists(MOTIONLCM_RENDER_SCRIPT_PATH):
        print("=" * 50)
        print(f"Error: render.py was not found: {os.path.abspath(MOTIONLCM_RENDER_SCRIPT_PATH)}")
        print("=" * 50)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    motion_name_base = (
        "".join(x for x in args.prompt if x.isalnum() or x in " _-")
        .strip().replace(" ", "_")
    )[:64] or "motion_output"

    base_output_dir = os.path.join(args.output_dir, motion_name_base)
    os.makedirs(base_output_dir, exist_ok=True)

    npy_path, total_samples_generated = step1_generate_skeleton(args, base_output_dir)

    print(f"\n=== Starting SMPL-X conversion for {total_samples_generated} samples ===")

    try:
        results_data = np.load(npy_path, allow_pickle=True).item()
        all_motions = results_data['motion']
        all_texts   = results_data['text']
        all_lengths = results_data['lengths']
    except Exception as e:
        print(f"Error: failed to load {npy_path}: {e}")
        sys.exit(1)

    for i in range(total_samples_generated):
        sample_name = f"sample_{i:02d}"
        print(f"\n--- Processing Sample {i+1}/{total_samples_generated}: {sample_name} ---")

        sample_output_dir = os.path.join(base_output_dir, sample_name)
        os.makedirs(sample_output_dir, exist_ok=True)

        use_cuda    = torch.cuda.is_available() and args.device != -1
        device      = args.device if use_cuda else 'cpu'

        # all_motions[i] shape: (njoints+1, 6, T)
        sample_hint = all_motions[i][-1, :3, :].T   # (T, 3)

        mesh_pkl_path = step2_convert_to_smplx_sequence(
            npy_path, i, 0,
            sample_output_dir, sample_name,
            device, use_cuda, sample_hint,
        )

        if mesh_pkl_path is None or not os.path.exists(mesh_pkl_path):
            print(f"  [ERROR] Mesh pkl not created. Skipping sample {i}.")
            continue

        print(f"  Mesh pkl created: [{mesh_pkl_path}]")

        print(f"  Running MotionLCM render.py with Blender...")
        render_command = (
            f"{BLENDER_PATH} --background --python {MOTIONLCM_RENDER_SCRIPT_PATH} -- "
            f"--pkl {mesh_pkl_path} "
            f"--mode video "
            f"--fps 20 "
            f"--faces_path {SMPLX_FACES_PATH}"
        )
        os.system(render_command)

        output_video_path = mesh_pkl_path.replace(".pkl", ".mp4")
        if os.path.exists(output_video_path):
            print(f"  Successfully rendered: [{output_video_path}]")
        else:
            print(f"  Warning: Rendered video not found at [{output_video_path}]")

    print(f"\n[SUCCESS] Pipeline finished for all {total_samples_generated} samples.")
    print(f"Results saved under: {os.path.abspath(base_output_dir)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        if 'NoSuchDisplayException' in str(e):
            print("\n======================= HINT =======================")
            print("Caught a 'NoSuchDisplayException'. Headless server detected.")
            print("\n1. Using EGL (Recommended):")
            print("   export PYOPENGL_PLATFORM=egl")
            print("   python motionlcm_render_smplx.py ...")
            print("\n2. Using a virtual display (Xvfb):")
            print("   xvfb-run python motionlcm_render_smplx.py ...")
            print("====================================================")
        else:
            raise e
