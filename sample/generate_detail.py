# smpl_visualization.py
# FINAL VERSION - Based on the batched generation logic (num_samples).
# ------------------------------------------------------------------------------------
# This script integrates three processes for 3D Human Motion Generation:
# 1. Text-to-Skeleton: Generates a BATCH of skeleton motions from a text prompt.
# 2. Skeleton-to-SMPL: Converts each skeleton data into a sequence of SMPL .obj meshes.
# 3. SMPL-to-Video: Renders each .obj sequence into a final 3D video.
#
# To Run:
# python smpl_visualization.py --model_path ./path/to/your/model.pt --prompt "a person is walking forward" --num_samples 10
# ------------------------------------------------------------------------------------

import os
import shutil
import argparse
import numpy as np
import torch
import trimesh
import pyrender
from tqdm import tqdm
import imageio
from PIL import Image, ImageDraw, ImageFont
import math
import sys

# --- [COMMON UTILITIES & SETUP] ---
from utils.fixseed import fixseed
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
from data_loaders.tensors import collate
from visualize import vis_utils

# =================================================================================================
# [STEP 3 LOGIC]: OBJ SEQUENCE RENDERING (No changes needed)
# =================================================================================================

large_scale = 1.8
height = 0.1
distance = 3.0

def look_at(camera_position, target, up=np.array([0, 1, 0])):
    z_axis = camera_position - target
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    view_matrix = np.eye(4)
    view_matrix[0, :3] = x_axis
    view_matrix[1, :3] = y_axis
    view_matrix[2, :3] = z_axis
    view_matrix[:3, 3] = camera_position
    return view_matrix

def load_obj_as_mesh(obj_path, color_index=0):
    trimesh_obj = trimesh.load(obj_path)
    trimesh_obj.apply_scale(large_scale)
    centroid = trimesh_obj.centroid
    color_gradient = [
        [1.0, 1.0, 0.6, 1.0], [1.0, 0.8, 0.4, 1.0], [1.0, 0.6, 0.2, 1.0],
        [1.0, 0.4, 0.0, 1.0], [0.8, 0.2, 0.0, 1.0], [0.6, 0.1, 0.0, 1.0],
        [0.4, 0.0, 0.0, 1.0]
    ]
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=color_gradient[color_index % len(color_gradient)],
        metallicFactor=0.0, roughnessFactor=0.5
    )
    pyrender_mesh = pyrender.Mesh.from_trimesh(trimesh_obj, material=material)
    return pyrender_mesh, centroid

def create_ground_plane(width=10.0, depth=10.0, y=0.0, color=[0.5, 0.5, 0.5, 1.0], center=np.array([0, 0, 0])):
    vertices = np.array([
        [-width / 2, y, -depth / 2], [width / 2, y, -depth / 2],
        [width / 2, y, depth / 2], [-width / 2, y, depth / 2]
    ]) + np.array([center[0], 0, center[2]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    plane = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=color, metallicFactor=0.0, roughnessFactor=1.0
    )
    return pyrender.Mesh.from_trimesh(plane, material=material)

def create_scene_for_render(mesh, model_center):
    scene = pyrender.Scene(bg_color=[0, 0, 0, 1])
    ground = create_ground_plane(center=np.array([model_center[0], 0, model_center[2]]))
    scene.add(ground)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    light_pose = np.array([
        [1, 0, 0, 0], [0, 1, 0, 2], [0, 0, 1, 2], [0, 0, 0, 1]
    ])
    scene.add(light, pose=light_pose)
    scene.add(mesh)
    return scene

def render_frame(scene, model_center):
    square_size = 1440
    renderer = pyrender.OffscreenRenderer(viewport_width=square_size, viewport_height=square_size)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 2.0, aspectRatio=1.0)
    offset = np.array([0, height, distance])
    camera_position = model_center + offset
    camera_pose = look_at(camera_position, model_center, up=np.array([0, 1, 0]))
    if scene.main_camera_node is not None:
        scene.remove_node(scene.main_camera_node)
    scene.add(camera, pose=camera_pose)
    color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()
    return color

def step3_render_final_video(obj_dir, output_dir, motion_name, fps=20):
    print("\n--- Step 3: Rendering OBJ sequence to final 3D video ---")
    output_image_dir = os.path.join(output_dir, f"{motion_name}_rendered_frames")
    output_video_path = os.path.join(output_dir, f"{motion_name}_rendered.mp4")

    os.makedirs(output_image_dir, exist_ok=True)
    obj_files = sorted([f for f in os.listdir(obj_dir) if f.endswith('.obj')])
    images = []

    for frame_idx, obj_file in enumerate(tqdm(obj_files, desc="Rendering Frames")):
        obj_path = os.path.join(obj_dir, obj_file)
        color_index = frame_idx // 28
        mesh, centroid = load_obj_as_mesh(obj_path, color_index)
        scene = create_scene_for_render(mesh, centroid)
        rendered_image = render_frame(scene, centroid)
        
        output_image_path = os.path.join(output_image_dir, obj_file.replace('.obj', '.png'))
        imageio.imwrite(output_image_path, rendered_image, format='PNG')
        images.append(rendered_image)
    
    print(f"Saving final rendered video to {output_video_path}...")
    imageio.mimsave(output_video_path, images, fps=fps, macro_block_size=1)
    
    print(f"Saved {len(images)} frames to [{output_image_dir}]")
    print(f"Saved final video to [{output_video_path}]")
    return output_video_path, output_image_dir

# =================================================================================================
# [STEP 2 LOGIC]: SKELETON TO SMPL .OBJ SEQUENCE (No changes needed)
# =================================================================================================

def step2_convert_to_smpl_sequence(npy_path, sample_i, rep_i, output_dir, motion_name, device, use_cuda):
    print("\n--- Step 2: Converting skeleton data to SMPL .obj sequence ---")
    
    obj_dir = os.path.join(output_dir, f"{motion_name}_obj")
    if os.path.exists(obj_dir):
        shutil.rmtree(obj_dir)
    os.makedirs(obj_dir)

    npy2obj = vis_utils.npy2obj(npy_path, sample_i, rep_i,
                                device=device, cuda=use_cuda)

    print(f"Saving .obj files to [{obj_dir}]")
    for frame_i in tqdm(range(npy2obj.real_num_frames), desc="Exporting .obj files"):
        npy2obj.save_obj(os.path.join(obj_dir, 'frame{:03d}.obj'.format(frame_i)), frame_i)
        
    out_smpl_params_path = os.path.join(output_dir, f"{motion_name}_smpl_params.npy")
    print(f'Saving SMPL params to [{out_smpl_params_path}]')
    npy2obj.save_npy(out_smpl_params_path)
    
    return obj_dir

# =================================================================================================
# [STEP 1 LOGIC]: TEXT TO SKELETON GENERATION (Updated to new logic)
# =================================================================================================

def step1_generate_skeleton(args, motion_name):
    """
    Generates a BATCH of skeleton motions from a text prompt.
    """
    print("--- Step 1: Generating skeleton motion from text ---")
    fixseed(args.seed)

    # [MODIFIED] Key logic: batch_size is the number of samples to generate.
    args.batch_size = args.num_samples
    
    out_path = os.path.join(args.output_dir, motion_name)
    
    max_frames = 196 if args.dataset in ['kit', 'humanml'] else 60
    fps = 12.5 if args.dataset == 'kit' else 20
    n_frames = min(max_frames, int(args.motion_length * fps))

    dist_util.setup_dist(args.device)
    
    print(f"Prompt: {args.prompt}")
    print(f"Number of samples to generate: {args.num_samples}")

    # Prepare texts for the batch
    texts = [args.prompt] * args.num_samples
    collate_args = [{'inp': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames}] * args.num_samples
    collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
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

    # The 'repetition' loop is for repeating the entire batch generation.
    # For our use case, we usually want num_repetitions = 1.
    all_motions = []
    all_lengths = []
    all_text = []

    for rep_i in range(args.num_repetitions):
        print(f'### Sampling [repetitions #{rep_i+1}/{args.num_repetitions}]')

        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
        
        sample_fn = diffusion.p_sample_loop
        sample = sample_fn(
            model,
            (args.batch_size, model.njoints, model.nfeats, n_frames), # The shape now uses batch_size
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
            sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
            sample = recover_from_ric(sample, n_joints)
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

        rot2xyz_pose_rep = 'xyz' if model.data_rep in ['xyz', 'hml_vec'] else model.data_rep
        rot2xyz_mask = None if rot2xyz_pose_rep == 'xyz' else model_kwargs['y']['mask'].reshape(args.batch_size, n_frames).bool()
        sample = model.rot2xyz(x=sample, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                               jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
                               get_rotations_back=False)
        
        all_motions.append(sample.cpu().numpy())
        all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())
        all_text.extend(texts)

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    all_motions = np.concatenate(all_motions, axis=0)
    all_lengths = np.concatenate(all_lengths, axis=0)

    npy_path = os.path.join(out_path, 'results.npy')
    print(f"Saving all {len(all_motions)} generated samples to [{npy_path}]")
    np.save(npy_path, {
        'motion': all_motions, 
        'text': all_text, 
        'lengths': all_lengths,
        'num_samples': args.num_samples,
        'num_repetitions': args.num_repetitions
    })
    
    total_samples = args.num_samples * args.num_repetitions
    return npy_path, total_samples

# =================================================================================================
# [MAIN ORCHESTRATOR]
# =================================================================================================

def main():
    # We use the original generate_args() and supplement it with our own arguments.
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument("--prompt", type=str, required=True, help="Text description of the desired motion.")
    # [MODIFIED] The primary control is num_samples, not num_repetitions.
    temp_parser.add_argument("--num_samples", type=int, default=10, help="Number of different samples to generate for the same prompt.")
    
    prompt_args, remaining_argv = temp_parser.parse_known_args()
    sys.argv[1:] = remaining_argv
    
    args = generate_args()

    args.prompt = prompt_args.prompt
    args.num_samples = prompt_args.num_samples
    # For this workflow, we almost always want to keep repetitions at 1.
    if args.num_repetitions > 1:
        print(f"Warning: num_repetitions is set to {args.num_repetitions}. The script will generate {args.num_samples} samples, {args.num_repetitions} times, for a total of {args.num_samples * args.num_repetitions} motions.")

    # --- Pipeline Execution ---
    os.makedirs(args.output_dir, exist_ok=True)
    
    motion_name_base = "".join(x for x in args.prompt if x.isalnum() or x in " _-").strip().replace(" ", "_")
    if not motion_name_base:
        motion_name_base = "motion_output"
    
    # This is the main directory for this prompt's results.
    base_output_dir = os.path.join(args.output_dir, motion_name_base)
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)

    # Step 1: Generate a BATCH of skeleton data
    npy_path, total_samples_generated = step1_generate_skeleton(args, base_output_dir)
    
    print(f"\n=== Starting to process {total_samples_generated} samples one by one ===")

    for i in range(total_samples_generated):
        # Each sample gets its own subdirectory for clarity.
        sample_name = f"sample_{i:02d}"
        print(f"\n--- Processing Sample {i+1}/{total_samples_generated}: {sample_name} ---")

        sample_output_dir = os.path.join(base_output_dir, sample_name)
        if not os.path.exists(sample_output_dir):
            os.makedirs(sample_output_dir)

        # Step 2: Convert skeleton data to SMPL obj sequence
        # The 'rep_i' argument for npy2obj is 0 because we handle repetitions outside.
        use_cuda = torch.cuda.is_available() and args.device != -1
        device = args.device if use_cuda else 'cpu'
        obj_dir = step2_convert_to_smpl_sequence(npy_path, i, 0, sample_output_dir, sample_name, device, use_cuda)

        # Step 3: Render the final 3D video from the obj sequence
        step3_render_final_video(obj_dir, sample_output_dir, sample_name, fps=20)
    
    print(f"\n[SUCCESS] Pipeline finished for all {total_samples_generated} samples.")
    print(f"All results are saved under the base directory: {os.path.abspath(base_output_dir)}")


if __name__ == "__main__":
    # Add a try-except block to catch the headless rendering error and provide a helpful message.
    try:
        main()
    except Exception as e:
        if 'NoSuchDisplayException' in str(e):
            print("\n======================= HINT =======================")
            print("Caught a 'NoSuchDisplayException'. This usually means you are on a headless server.")
            print("Try running the script with one of the following prefixes:")
            print("\n1. Using EGL (Recommended):")
            print("   export PYOPENGL_PLATFORM=egl")
            print("   python smpl_visualization.py ...")
            print("\n2. Using a virtual display (Xvfb):")
            print("   xvfb-run python smpl_visualization.py ...")
            print("====================================================")
        else:
            raise e