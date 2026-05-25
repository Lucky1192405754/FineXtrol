# # motionlcm_render.py
# # ------------------------------------------------------------------------------------

# import os
# import shutil
# import argparse
# import numpy as np
# import torch
# import sys
# import pickle 
# import random
# import json

# import trimesh
# from tqdm import tqdm

# from utils.fixseed import fixseed
# from utils.parser_util import generate_args
# from utils.model_util import create_model_and_diffusion, load_model_wo_clip
# from utils import dist_util
# from model.cfg_sampler import ClassifierFreeSampleModel
# from data_loaders.get_data import get_dataset_loader
# from data_loaders.humanml.scripts.motion_process import recover_from_ric
# import data_loaders.humanml.utils.paramUtil as paramUtil
# from data_loaders.humanml.utils.plot_script import plot_3d_motion
# from data_loaders.tensors import collate
# from visualize import vis_utils 

# # (pyrender, imageio, PIL, math)
# # ------------------------

# # =================================================================================================
# # [STEP 4 LOGIC]: DETAILED TEXT PROCESSING HELPER
# # =================================================================================================
# def _process_detailed_text(dt, selected_joints, mask_ratio):
#     """
#     """
#     processed_sentences = []

#     for sentence in dt:
#         if not sentence or sentence.strip() == "":
#             processed_sentences.append("<Motionless>")
#         else:
#             min_units = [s.strip() for s in sentence.split('.') if s.strip()]
#             filtered_units = [unit for unit in min_units if any(joint in unit for joint in selected_joints)]
            
#             if filtered_units:
#                 processed_sentence = ". ".join(filtered_units) + "."
#                 processed_sentences.append(processed_sentence)
#             else:
#                 processed_sentences.append("<Mask>")


#     for i in range(len(processed_sentences)):
#         current_token = processed_sentences[i]
        
#         if current_token != "<Mask>":
#             if random.random() < mask_ratio: 
#                 processed_sentences[i] = "<Mask>"

#     return processed_sentences

# # =================================================================================================
# # [STEP 3 LOGIC]: (REMOVED)
# # =================================================================================================
# # ------------------------


# # =================================================================================================
# # [STEP 2 LOGIC]: (RESTORED & MODIFIED)
# # =================================================================================================
# def step2_convert_to_smpl_sequence(npy_path, sample_i, rep_i, output_dir, motion_name, device, use_cuda, hint_data):
#     """
#     """
#     print(f"\n--- Step 2: Converting skeleton data to SMPL Mesh PKL (using vis_utils) ---")
    
#     try:
#         npy2obj = vis_utils.npy2obj(npy_path, sample_i, rep_i,
#                                     device=device, cuda=use_cuda)
#     except Exception as e:
#         print(f"  [ERROR] Error initializing npy2obj (vis_utils): {e}")
#         return None

#     all_mesh_vertices = []
    
#     print(f"  Extracting all mesh vertices from {npy2obj.real_num_frames} frames...")
    
#     temp_obj_path = os.path.join(output_dir, f"{motion_name}_temp_frame.obj")
    
#     for frame_i in tqdm(range(npy2obj.real_num_frames), desc="Extracting Frames"):
#         try:
#             npy2obj.save_obj(temp_obj_path, frame_i)
            
#             mesh = trimesh.load(temp_obj_path, process=False)
#             all_mesh_vertices.append(mesh.vertices)
            
#         except Exception as e:
#             print(f"  [ERROR] Error processing frame {frame_i}: {e}")
#             if os.path.exists(temp_obj_path):
#             return None
    
#     if os.path.exists(temp_obj_path):

#     try:
#         stacked_meshes = np.stack(all_mesh_vertices, axis=0) # Shape: (nframes, 6890, 3)
#     except Exception as e:
#         print(f"  [ERROR] Error stacking meshes. Do all meshes have the same vertex count? {e}")
#         return None

#     mesh_pkl_path = os.path.join(output_dir, f"{motion_name}_joints_data_mesh.pkl")
    
#     print(f'  Saving stacked SMPL meshes to [{mesh_pkl_path}]')
    
#     try:
#         data_to_save = {
#             'vertices': stacked_meshes, # (nframes, 6890, 3)
#             'hint': hint_data, # (nframes, 3)
#         }
        
#         with open(mesh_pkl_path, 'wb') as f:
#             pickle.dump(data_to_save, f)
            
#     except Exception as e:
#         print(f"  [ERROR] Error saving final mesh pkl: {e}")
#         return None

#     return mesh_pkl_path

# # =================================================================================================
# # [STEP 1 LOGIC]: TEXT TO SKELETON GENERATION (No changes)
# # =================================================================================================
# def step1_generate_skeleton(args, motion_name, n_frames):
#     """
#     """
#     print("--- Step 1: Generating skeleton motion from text ---")
#     fixseed(args.seed)

#     args.batch_size = args.num_samples
#     out_path = os.path.join(args.output_dir, motion_name)
#     max_frames = 196 if args.dataset in ['kit', 'humanml', 'detailed_text'] else 60
#     fps = 12.5 if args.dataset == 'kit' else 20
    
    
#     print(f"Motion length set to: {n_frames} frames.")

#     dist_util.setup_dist(args.device)
    
#     print(f"Coarse-grained prompt: {args.prompt}") 
#     print(f"Number of samples to generate: {args.num_samples}")

#     texts = [args.prompt] * args.num_samples
    
#     if args.detailed_text.strip() != '':
#         print(f"Fine-grained prompt: {args.detailed_text}") 
#         detailed_texts = [args.detailed_text] * args.num_samples
#     else:
#         default_dt = '<SEP> <Motionless> <SEP> <SEP> <Motionless> <SEP> Raise your right arm over your head and kick your left leg. <SEP> <Motionless> <SEP> <Motionless> <SEP>'
#         print(f"Fine-grained prompt not provided, using default.")
#         detailed_texts = [default_dt] * args.num_samples

#     collate_args = [{'inp': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames}] * args.num_samples
#     collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
#     collate_args = [dict(arg, detailed_text=dt) for arg, dt in zip(collate_args, detailed_texts)]

#     _, model_kwargs = collate(collate_args)
    
#     print("Loading dataset metadata...")
#     data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=max_frames)

#     print("Creating model and diffusion...") 
#     model, diffusion = create_model_and_diffusion(args, data)

#     print(f"Loading checkpoints from [{args.model_path}]...")
#     state_dict = torch.load(args.model_path, map_location='cpu')
#     load_model_wo_clip(model, state_dict)

#     if args.guidance_param != 1:
#         model = ClassifierFreeSampleModel(model)
#     model.to(dist_util.dev())
#     model.eval()

#     for k, v in model_kwargs['y'].items():
#         if torch.is_tensor(v):
#             model_kwargs['y'][k] = v.to(dist_util.dev())

#     all_motions = []
#     all_lengths = []
#     all_text = []
#     all_detailed_text = [] 

#     for rep_i in range(args.num_repetitions):
#         print(f'### Sampling [repetitions #{rep_i+1}/{args.num_repetitions}]')
#         if args.guidance_param != 1:
#             model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
        
#         sample_fn = diffusion.p_sample_loop
#         sample = sample_fn(model, (args.batch_size, model.njoints, model.nfeats, n_frames),
#                            clip_denoised=False, model_kwargs=model_kwargs, skip_timesteps=0, 
#                            init_image=None, progress=True, dump_steps=None, noise=None, const_noise=False) 

#         if model.data_rep == 'hml_vec':
#             n_joints = 22 if sample.shape[1] == 263 else 21
#             sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
#             sample = recover_from_ric(sample, n_joints) 
#             sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

#         rot2xyz_pose_rep = 'xyz' if model.data_rep in ['xyz', 'hml_vec'] else model.data_rep
#         rot2xyz_mask = None if rot2xyz_pose_rep == 'xyz' else model_kwargs['y']['mask'].reshape(args.batch_size, n_frames).bool()
#         sample = model.rot2xyz(x=sample, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
#                                jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None, 
#                                get_rotations_back=False) 
        
#         all_motions.append(sample.cpu().numpy())
#         all_lengths.append(np.array([n_frames] * args.batch_size))
#         all_text.extend(texts)
#         all_detailed_text.extend(detailed_texts)

#     if os.path.exists(out_path):
#         shutil.rmtree(out_path) 
#     os.makedirs(out_path)

#     all_motions = np.concatenate(all_motions, axis=0)
#     all_lengths = np.concatenate(all_lengths, axis=0)

#     npy_path = os.path.join(out_path, 'results.npy')
#     print(f"Saving all {len(all_motions)} generated samples to [{npy_path}]")
#     np.save(npy_path, {
#         'motion': all_motions, 
#         'text': all_text, 
#         'lengths': all_lengths,
#         'detailed_text': all_detailed_text,
#         'num_samples': args.num_samples,
#         'num_repetitions': args.num_repetitions
#     }) 
    
#     total_samples = args.num_samples * args.num_repetitions
#     return npy_path, total_samples


# # =================================================================================================
# # [MAIN ORCHESTRATOR] (Modified)
# # =================================================================================================
# def main():
#     temp_parser = argparse.ArgumentParser(add_help=False)
#     temp_parser.add_argument("--prompt", type=str, default=None, help="Coarse-grained text. If not given, --sample_id or random sampling will be used.")
#     temp_parser.add_argument("--detailed_text", type=str, default=None, help="Optional: Fine-grained text. If not given, default will be used.")
#     temp_parser.add_argument("--num_samples", type=int, default=2, help="Number of different samples to generate for the same prompt.")
#     temp_parser.add_argument("--sample_id", type=str, default=None, help="Specify a 6-digit HumanML3D ID. If not given, a random ID will be picked from test.txt.")
#     temp_parser.add_argument("--num_sequence_frames", type=int, default=10, help="Number of frames for the sequence image.")
#     temp_parser.add_argument("--num_test_ids", type=int, default=30, help="Number of different test IDs to sample from HumanML3D. (Default: 1)")
#     prompt_args, remaining_argv = temp_parser.parse_known_args()
    
#     sys.argv[1:] = remaining_argv 
    
#     args = generate_args()



#     args.prompt = prompt_args.prompt
#     args.detailed_text = prompt_args.detailed_text
#     args.num_samples = prompt_args.num_samples
#     args.sample_id = prompt_args.sample_id
#     args.num_sequence_frames = prompt_args.num_sequence_frames
#     args.num_test_ids = prompt_args.num_test_ids

#     dataset_base_path = "./dataset/HumanML3D"
#     test_txt_path = os.path.join(dataset_base_path, "test.txt")
#     texts_dir = os.path.join(dataset_base_path, "texts")
#     joints_dir = os.path.join(dataset_base_path, "new_joints")
#     detailed_text_json_path = "./dataset/0121_operated_mirror_ori_humanml3d_posefix_annotations_interval0.5_pose_change_th1.0_modified.json"
#     mask_ratio = 0.5
    
#     body_parts_dict = {
#         "head": ['head'],
#         "body": ['body', 'torso', 'waist', 'upper back', 'lower back'],
#         "left hand": ['left hand', 'left arm', 'left elbow', 'left shoulder', 'left forearm', 'hands', 'arms', 'elbows', 'shoulders', 'forearms'],
#         "right hand": ['right hand', 'right arm', 'right elbow', 'right shoulder', 'right forearm', 'hands', 'arms', 'elbows', 'shoulders', 'forearms'],
#         "left leg": ['left leg', 'left foot', 'left knee', 'left heel', 'legs', 'feet', 'knees', 'heels'],
#         "right leg": ['right leg', 'right foot', 'right knee', 'right heel', 'legs', 'feet', 'knees', 'heels']
#     } 

#     try:
#         with open(detailed_text_json_path, 'r') as f:
#             detailed_text_data = json.load(f) 
#     except FileNotFoundError:
#         print(f"ERROR: Cannot find detailed text JSON: {detailed_text_json_path}")
#         sys.exit(1)

#     all_test_ids = []
#     try:
#         with open(test_txt_path, 'r') as f:
#             all_test_ids = f.read().splitlines() 
#     except FileNotFoundError:
#         print(f"ERROR: Cannot find test.txt at {test_txt_path}")
#         sys.exit(1)

#     num_test_ids_to_run = args.num_test_ids
#     if args.sample_id is not None or args.prompt is not None:
#         if args.num_test_ids > 1:
#             print(f"Warning: --sample_id or --prompt was provided, forcing --num_test_ids to 1.")
#         num_test_ids_to_run = 1

#     for test_run_index in range(num_test_ids_to_run):
#         print("\n" + "#"*80)
#         print(f"### Starting Test Run {test_run_index + 1} / {num_test_ids_to_run} ###")
#         print("#"*80 + "\n")
        
#         selected_id = args.sample_id
#         current_prompt = args.prompt
#         current_detailed_text = args.detailed_text
#         n_frames = 196
        
#         if selected_id is None and current_prompt is None:
#             selected_id = random.choice(all_test_ids)
#             print(f"Randomly selected ID: {selected_id}")

#         if selected_id is not None:
#             print(f"Using HumanML3D sample ID: {selected_id}")
            
#             text_file_path = os.path.join(texts_dir, f"{selected_id}.txt")
#             try:
#                 with open(text_file_path, 'r') as f:
#                     all_prompts = f.read().splitlines()
#                 selected_prompt_line = random.choice(all_prompts)
#                 current_prompt = selected_prompt_line.split('#')[0].strip() 
#             except FileNotFoundError:
#                 print(f"ERROR: Cannot find text file: {text_file_path}")

#             joint_file_path = os.path.join(joints_dir, f"{selected_id}.npy")
#             try:
#                 joint_data = np.load(joint_file_path)
#                 n_frames = joint_data.shape[0] 
#             except FileNotFoundError:
#                 print(f"ERROR: Cannot find joints file: {joint_file_path}")
            
#             try:
#                 raw_dt_list = detailed_text_data[selected_id]
#                 selected_category = random.choice(list(body_parts_dict.keys()))
#                 selected_joints = body_parts_dict[selected_category]
#                 print(f"Randomly selected body part for processing: '{selected_category}'")
#                 processed_sentences = _process_detailed_text(raw_dt_list, selected_joints, mask_ratio) 
#                 current_detailed_text = " <SEP> ".join(processed_sentences) 
#             except KeyError:
#                 print(f"WARNING: No detailed text found for ID {selected_id} in JSON. Using default.")
#                 current_detailed_text = '' 

#         elif current_prompt is not None:
#             print(f"Using manually provided prompt.")
#             if current_detailed_text is None:
#                 current_detailed_text = '' 
        
#         else:
#             print("ERROR: Could not determine a prompt. Skipping run.")
#             continue

#         args.prompt = current_prompt
#         args.detailed_text = current_detailed_text
        
#         if args.num_repetitions > 1:
#             print(f"Warning: num_repetitions... (This setting applies per sample)") 

#         motion_name_base = "".join(x for x in args.prompt if x.isalnum() or x in " _-").strip().replace(" ", "_")[:50]
#         if not motion_name_base: motion_name_base = f"motion_{selected_id}"
        
#         base_output_dir = os.path.join(args.output_dir, motion_name_base)
#         if not os.path.exists(base_output_dir): 
#             os.makedirs(base_output_dir)
#         else:
#             if args.sample_id is None and args.prompt is None:
#                 base_output_dir = os.path.join(args.output_dir, f"{motion_name_base}_{selected_id}")
#                 if not os.path.exists(base_output_dir): os.makedirs(base_output_dir)

#         log_file_path = os.path.join(base_output_dir, "log.txt") 

#         print("\n" + "="*50)
#         print("      DATA SAMPLING VERIFICATION")
#         print("="*50)
#         log_content_lines = []
#         log_content_lines.append("="*50)
#         log_content_lines.append("      DATA SAMPLING VERIFICATION")
#         log_content_lines.append("="*50)

#         if selected_id:
#             log_line = f"Sample ID:      {selected_id}"
#             print(log_line)
#             log_content_lines.append(log_line)
            
#         log_line = f"Coarse Text:    {args.prompt}"
#         print(log_line)
#         log_content_lines.append(log_line)
        
#         log_line = f"Frame Length:   {n_frames}"
#         print(log_line)
#         log_content_lines.append(log_line)

#         if args.detailed_text.strip() == '':
#             log_line = "Detailed Text:  [Not provided, will use default in Step 1]"
#             print(log_line)
#             log_content_lines.append(log_line)
#         else:
#             log_content_lines.append(f"Detailed Text:  {args.detailed_text}")
            
#             dt_preview = args.detailed_text[:150] + "..." if len(args.detailed_text) > 150 else args.detailed_text
#             print(f"Detailed Text:  {dt_preview}")

#         print("="*50 + "\n")
#         log_content_lines.append("="*50 + "\n")

#         try:
#             with open(log_file_path, 'w') as log_f:
#                 log_f.write("\n".join(log_content_lines))
#             print(f"--- Log file saved to: {log_file_path} ---\n")
#         except Exception as e:
#             print(f"--- WARNING: Could not write log file. Error: {e} ---\n")

#         BLENDER_PATH = "/data/shenkeming/blender-3.6.23-linux-x64/blender" 
#         base_motionlcm_dir = "./Motionlcm" 
#         MOTIONLCM_RENDER_SCRIPT_PATH = os.path.join(base_motionlcm_dir, "render.py") 

#         if not os.path.exists(BLENDER_PATH):
#         if not os.path.exists(MOTIONLCM_RENDER_SCRIPT_PATH):

#         npy_path, total_samples_generated = step1_generate_skeleton(args, base_output_dir, n_frames) 
        
#         print(f"\n=== Starting to process {total_samples_generated} samples for this ID ===")

#         try:
#             results_data = np.load(npy_path, allow_pickle=True).item()
#             all_motions = results_data['motion']
#             all_texts = results_data['text']
#             all_lengths = results_data['lengths']
#         except Exception as e:

#         for i in range(total_samples_generated):
#             sample_name = f"sample_{i:02d}"
#             print(f"\n--- Processing Sample {i+1}/{total_samples_generated}: {sample_name} ---") 
#             sample_output_dir = os.path.join(base_output_dir, sample_name)
#             if not os.path.exists(sample_output_dir): os.makedirs(sample_output_dir) 

#             use_cuda = torch.cuda.is_available() and args.device != -1
#             device = args.device if use_cuda else 'cpu' 

#             sample_motion = all_motions[i]
#             sample_hint = sample_motion[:, 0, :] 

#             mesh_pkl_path = step2_convert_to_smpl_sequence(
#                 npy_path, i, 0, sample_output_dir, sample_name, 
#                 device, use_cuda, sample_hint 
#             ) 

#             if mesh_pkl_path is None or not os.path.exists(mesh_pkl_path):
#                 print(f"  [ERROR] Mesh file {mesh_pkl_path} was not created by local Step 2.")
#                 continue 
            
#             print(f"  Mesh file created: [{mesh_pkl_path}]")

#             print(f"  Running MotionLCM render.py with Blender...(Mode: video)")
#             render_command = (
#                 f"{BLENDER_PATH} --background --python {MOTIONLCM_RENDER_SCRIPT_PATH} -- "
#                 f"--pkl {mesh_pkl_path} --mode video --fps 20"
#             ) 
#             os.system(render_command)
#             output_video_path = mesh_pkl_path.replace(".pkl", ".mp4")
#             if os.path.exists(output_video_path):
#                 print(f"  Successfully rendered video to [{output_video_path}]")
#             else:
#                 print(f"  Warning: Rendered video not found at [{output_video_path}]") 
            
#             print(f"  Running MotionLCM render.py with Blender... (Mode: sequence)")
#             render_command_seq = (
#                 f"{BLENDER_PATH} --background --python {MOTIONLCM_RENDER_SCRIPT_PATH} -- "
#                 f"--pkl {mesh_pkl_path} --mode sequence --num {args.num_sequence_frames}"
#             ) 
#             os.system(render_command_seq)
#             output_seq_path = mesh_pkl_path.replace(".pkl", ".png")
#             if os.path.exists(output_seq_path):
#                 print(f"  Successfully rendered sequence to [{output_seq_path}]")
#             else:
#                 print(f"  Warning: Rendered sequence not found at [{output_seq_path}]")
    
#     print(f"\n[SUCCESS] All {num_test_ids_to_run} test runs finished.") 



# if __name__ == "__main__":
#     try:
#         main()
#     except Exception as e:
#         if 'NoSuchDisplayException' in str(e):
#              print("\n======================= HINT =======================")
#              print("Caught a 'NoSuchDisplayException'. This usually means you are on a headless server.") 
#              print("\n1. Using EGL (Recommended):")
#              print("   export PYOPENGL_PLATFORM=egl")
#              print("   python motionlcm_render.py ...")
#              print("\n2. Using a virtual display (Xvfb):")
#              print("   xvfb-run python motionlcm_render.py ...")
#              print("====================================================") 
#         else:
#             raise e


# motionlcm_render.py
# ------------------------------------------------------------------------------------

import os
import shutil
import argparse
import numpy as np
import torch
import sys
import pickle 
import random
import json

import trimesh
from tqdm import tqdm

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

# (pyrender, imageio, PIL, math)
# ------------------------

# =================================================================================================
# [STEP 4 LOGIC]: DETAILED TEXT PROCESSING HELPER
# =================================================================================================
def _process_detailed_text(dt, selected_joints, mask_ratio):
    """
    [Fixed]
    Process fine-grained text data, adapted from Text2MotionDetailedText.
    :param dt: Raw fine-grained text as a list of sentences. 
    :param selected_joints: Joint list for the selected body part. 
    :param mask_ratio: Mask ratio. 
    :return: Processed sentence list.
    """
    processed_sentences = []

    for sentence in dt:
        if not sentence or sentence.strip() == "":
            processed_sentences.append("<Motionless>")
        else:
            min_units = [s.strip() for s in sentence.split('.') if s.strip()]
            filtered_units = [unit for unit in min_units if any(joint in unit for joint in selected_joints)]
            
            if filtered_units:
                processed_sentence = ". ".join(filtered_units) + "."
                processed_sentences.append(processed_sentence)
            else:
                processed_sentences.append("<Mask>")


    for i in range(len(processed_sentences)):
        current_token = processed_sentences[i]
        
        if current_token != "<Mask>" and current_token != "<Motionless>":
            if random.random() < mask_ratio: 
                processed_sentences[i] = "<Mask>"

    return processed_sentences

# =================================================================================================
# [STEP 3 LOGIC]: (REMOVED)
# =================================================================================================
# ------------------------


# =================================================================================================
# [STEP 2 LOGIC]: (RESTORED & MODIFIED)
# =================================================================================================
def step2_convert_to_smpl_sequence(npy_path, sample_i, rep_i, output_dir, motion_name, device, use_cuda, hint_data):
    """
    [Modified]
    This function now uses vis_utils (npy2obj) to:
    1. Load the i-th skeleton generated by the model.
    2. Iterate over frames and obtain SMPL mesh vertices.
    3. Stack vertices from all frames.
    4. Save the (nframes, 6890, 3) mesh and (nframes, 3) trajectory (hint_data)
       as a single .pkl file for MotionLCM/render.py.
    """
    print(f"\n--- Step 2: Converting skeleton data to SMPL Mesh PKL (using vis_utils) ---")
    
    try:
        npy2obj = vis_utils.npy2obj(npy_path, sample_i, rep_i,
                                    device=device, cuda=use_cuda)
    except Exception as e:
        print(f"  [ERROR] Error initializing npy2obj (vis_utils): {e}")
        print("  [ERROR] Please make sure 'visualize.vis_utils' exists and is compatible with your environment.")
        return None

    all_mesh_vertices = []
    
    print(f"  Extracting all mesh vertices from {npy2obj.real_num_frames} frames...")
    
    temp_obj_path = os.path.join(output_dir, f"{motion_name}_temp_frame.obj")
    
    for frame_i in tqdm(range(npy2obj.real_num_frames), desc="Extracting Frames"):
        try:
            npy2obj.save_obj(temp_obj_path, frame_i)
            
            mesh = trimesh.load(temp_obj_path, process=False)
            all_mesh_vertices.append(mesh.vertices)
            
        except Exception as e:
            print(f"  [ERROR] Error processing frame {frame_i}: {e}")
            if os.path.exists(temp_obj_path):
                os.remove(temp_obj_path)
            return None
    
    if os.path.exists(temp_obj_path):
        os.remove(temp_obj_path)

    try:
        stacked_meshes = np.stack(all_mesh_vertices, axis=0) # Shape: (nframes, 6890, 3)
    except Exception as e:
        print(f"  [ERROR] Error stacking meshes. Do all meshes have the same vertex count? {e}")
        return None

    mesh_pkl_path = os.path.join(output_dir, f"{motion_name}_joints_data_mesh.pkl")
    
    print(f'  Saving stacked SMPL meshes to [{mesh_pkl_path}]')
    
    try:
        data_to_save = {
            'vertices': stacked_meshes, # (nframes, 6890, 3)
            'hint': hint_data, # (nframes, 3)
        }
        
        with open(mesh_pkl_path, 'wb') as f:
            pickle.dump(data_to_save, f)
            
    except Exception as e:
        print(f"  [ERROR] Error saving final mesh pkl: {e}")
        return None

    return mesh_pkl_path

# =================================================================================================
# [STEP 1 LOGIC]: TEXT TO SKELETON GENERATION (No changes)
# =================================================================================================
def step1_generate_skeleton(args, motion_name, n_frames, current_prompt, current_detailed_text):
    """
    (This function is unchanged and correctly handles detailed_text.) 
    """
    print("--- Step 1: Generating skeleton motion from text ---")
    fixseed(args.seed)

    args.batch_size = args.num_samples
    out_path = os.path.join(args.output_dir, motion_name)
    max_frames = 196 if args.dataset in ['kit', 'humanml', 'detailed_text'] else 60
    fps = 12.5 if args.dataset == 'kit' else 20
    
    print(f"Motion length set to: {n_frames} frames.") 

    dist_util.setup_dist(args.device)
    
    print(f"Coarse-grained prompt: {current_prompt}") 
    print(f"Number of samples to generate: {args.num_samples}")

    texts = [current_prompt] * args.num_samples
    
    if current_detailed_text.strip() != '':
        print(f"Fine-grained prompt: {current_detailed_text}") 
        detailed_texts = [current_detailed_text] * args.num_samples
    else:
        default_dt = '<SEP> <Motionless> <SEP> <SEP> <Motionless> <SEP> Raise your right arm over your head and kick your left leg. <SEP> <Motionless> <SEP> <Motionless> <SEP>' 
        print(f"Fine-grained prompt not provided, using default.")
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

    print(f'### Sampling [batch size #{args.num_samples}]')
    if args.guidance_param != 1:
        model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
    
    sample_fn = diffusion.p_sample_loop
    sample = sample_fn(model, (args.batch_size, model.njoints, model.nfeats, n_frames),
                       clip_denoised=False, model_kwargs=model_kwargs, skip_timesteps=0, 
                       init_image=None, progress=True, dump_steps=None, 
                       noise=None, const_noise=False) 

    if model.data_rep == 'hml_vec':
        n_joints = 22 if sample.shape[1] == 263 else 21
        sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
        sample = recover_from_ric(sample, n_joints) 
        sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1) 

    rot2xyz_pose_rep = 'xyz' if model.data_rep in ['xyz', 'hml_vec'] else model.data_rep
    rot2xyz_mask = None if rot2xyz_pose_rep == 'xyz' else model_kwargs['y']['mask'].reshape(args.batch_size, n_frames).bool()
    all_motions = model.rot2xyz(x=sample, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                           jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None, 
                           get_rotations_back=False) 
    
    all_lengths = np.array([n_frames] * args.batch_size)
    all_text = texts
    all_detailed_text = detailed_texts

    if os.path.exists(out_path):
         shutil.rmtree(out_path) 
    os.makedirs(out_path) 

    
    npy_path = os.path.join(out_path, 'results.npy')
    print(f"Saving all {len(all_motions)} generated samples to [{npy_path}]")
    np.save(npy_path, {
        'motion': all_motions.cpu().numpy(),
        'text': all_text, 
        'lengths': all_lengths,
        'detailed_text': all_detailed_text,
        'num_samples': args.num_samples,
        'num_repetitions': 1
    }) 
    
    total_samples = args.num_samples
    return npy_path, total_samples


# =================================================================================================
# [MAIN ORCHESTRATOR] (Modified)
# =================================================================================================
def main():
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument("--prompt", type=str, default=None, help="Coarse-grained text. If not given, --sample_id or random sampling will be used.")
    temp_parser.add_argument("--detailed_text", type=str, default=None, help="Optional: Fine-grained text. If not given, default will be used.")
    temp_parser.add_argument("--num_samples", type=int, default=2, help="Number of different samples to generate for the same prompt.")
    temp_parser.add_argument("--sample_id", type=str, default=None, help="Specify a 6-digit HumanML3D ID. If not given, a random ID will be picked from test.txt.")
    temp_parser.add_argument("--num_sequence_frames", type=int, default=8, help="Number of frames for the sequence image.")
    temp_parser.add_argument("--num_test_ids", type=int, default=30, help="Number of different test IDs to sample from HumanML3D. (Default: 1)")
    temp_parser.add_argument("--body_part", type=str, nargs='+', default=None, 
                             choices=['head', 'body', 'left hand', 'right hand', 'left leg', 'right leg'], 
                             help="Controlling part for FineXtrol.")
    prompt_args, remaining_argv = temp_parser.parse_known_args()
    
    sys.argv[1:] = remaining_argv 
    
    args = generate_args()



    args.prompt = prompt_args.prompt
    args.detailed_text = prompt_args.detailed_text
    args.num_samples = prompt_args.num_samples
    args.sample_id = prompt_args.sample_id
    args.num_sequence_frames = prompt_args.num_sequence_frames
    args.num_test_ids = prompt_args.num_test_ids
    args.body_part = prompt_args.body_part

    dataset_base_path = "./dataset/HumanML3D"
    test_txt_path = os.path.join(dataset_base_path, "test.txt")
    texts_dir = os.path.join(dataset_base_path, "texts")
    joints_dir = os.path.join(dataset_base_path, "new_joints")
    detailed_text_json_path = "./dataset/0121_operated_mirror_ori_humanml3d_posefix_annotations_interval0.5_pose_change_th1.0_modified.json"
    mask_ratio = 0.5
    
    body_parts_dict = {
        "head": ['head'],
        "body": ['body', 'torso', 'waist', 'upper back', 'lower back'],
        "left hand": ['left hand', 'left arm', 'left elbow', 'left shoulder', 'left forearm', 'hands', 'arms', 'elbows', 'shoulders', 'forearms'],
        "right hand": ['right hand', 'right arm', 'right elbow', 'right shoulder', 'right forearm', 'hands', 'arms', 'elbows', 'shoulders', 'forearms'],
        "left leg": ['left leg', 'left foot', 'left knee', 'left heel', 'legs', 'feet', 'knees', 'heels'],
        "right leg": ['right leg', 'right foot', 'right knee', 'right heel', 'legs', 'feet', 'knees', 'heels']
    } 

    try:
        with open(detailed_text_json_path, 'r') as f:
            detailed_text_data = json.load(f) 
    except FileNotFoundError:
        print(f"ERROR: Cannot find detailed text JSON: {detailed_text_json_path}")
        sys.exit(1)

    all_test_ids = []
    try:
        with open(test_txt_path, 'r') as f:
            all_test_ids = f.read().splitlines() 
    except FileNotFoundError:
        print(f"ERROR: Cannot find test.txt at {test_txt_path}")
        sys.exit(1)

    num_test_ids_to_run = args.num_test_ids
    if args.sample_id is not None or args.prompt is not None:
        if args.num_test_ids > 1:
            print(f"Warning: --sample_id or --prompt was provided, forcing --num_test_ids to 1.")
        num_test_ids_to_run = 1

    
    for test_run_index in range(num_test_ids_to_run):
        print("\n" + "#"*80)
        print(f"### Starting Test Run {test_run_index + 1} / {num_test_ids_to_run} ###")
        print("#"*80 + "\n")
         
        selected_id = args.sample_id
        
        if args.sample_id is None and args.prompt is None:
            current_prompt = None
            current_detailed_text = None
        else:
            current_prompt = args.prompt
            current_detailed_text = args.detailed_text

        n_frames = 196
        
        if selected_id is None and current_prompt is None:
            selected_id = random.choice(all_test_ids) 
            print(f"Randomly selected ID: {selected_id}")

        if selected_id is not None:
            print(f"Using HumanML3D sample ID: {selected_id}")
            
            text_file_path = os.path.join(texts_dir, f"{selected_id}.txt") 
            try:
                with open(text_file_path, 'r') as f:
                    all_prompts = f.read().splitlines()
                selected_prompt_line = random.choice(all_prompts)
                current_prompt = selected_prompt_line.split('#')[0].strip() 
            except FileNotFoundError:
                print(f"ERROR: Cannot find text file: {text_file_path}")
                continue

            joint_file_path = os.path.join(joints_dir, f"{selected_id}.npy") 
            try:
                joint_data = np.load(joint_file_path)
                n_frames = joint_data.shape[0] 
            except FileNotFoundError:
                print(f"ERROR: Cannot find joints file: {joint_file_path}")
                continue
            
            try:
                raw_dt_list = detailed_text_data[selected_id]
                
                selected_categories_list = args.body_part
                
                if selected_categories_list is None:
                    selected_category = random.choice(list(body_parts_dict.keys()))
                    selected_categories_list = [selected_category] 
                    print(f"Randomly selected body part for processing: '{selected_category}'")
                else:
                    print(f"Using user-specified body part(s): {selected_categories_list}")

                selected_joints = []
                for category in selected_categories_list:
                    selected_joints.extend(body_parts_dict[category])
                selected_joints = list(set(selected_joints))
                
                processed_sentences = _process_detailed_text(raw_dt_list, selected_joints, mask_ratio) 
                current_detailed_text = " <SEP> ".join(processed_sentences) 
            except KeyError:
                print(f"WARNING: No detailed text found for ID {selected_id} in JSON. Using default.")
                current_detailed_text = '' 

        elif current_prompt is not None:
            print(f"Using manually provided prompt.")
            if current_detailed_text is None:
                current_detailed_text = '' 
        
        else:
            print("ERROR: Could not determine a prompt. Skipping run.")
            continue

        
        if args.num_repetitions > 1:
             print(f"Warning: num_repetitions... (This setting applies per sample)") 

        motion_name_base = "".join(x for x in current_prompt if x.isalnum() or x in " _-").strip().replace(" ", "_")[:50]
        if not motion_name_base: motion_name_base = f"motion_{selected_id}"
        
        base_output_dir = os.path.join(args.output_dir, motion_name_base)
        if not os.path.exists(base_output_dir): 
           os.makedirs(base_output_dir) 
        else:
            if args.sample_id is None and args.prompt is None:
                base_output_dir = os.path.join(args.output_dir, f"{motion_name_base}_{selected_id}")
                if not os.path.exists(base_output_dir): os.makedirs(base_output_dir)

        log_file_path = os.path.join(base_output_dir, "log.txt") 

        print("\n" + "="*50)
        print("      DATA SAMPLING VERIFICATION")
        print("="*50)
        log_content_lines = []
        log_content_lines.append("="*50)
        log_content_lines.append("      DATA SAMPLING VERIFICATION")
        log_content_lines.append("="*50)

        if selected_id:
           log_line = f"Sample ID:      {selected_id}"
           print(log_line)
           log_content_lines.append(log_line)
            
        log_line = f"Coarse Text:    {current_prompt}"
        print(log_line)
        log_content_lines.append(log_line)
        
        log_line = f"Frame Length:   {n_frames}"
        print(log_line)
        log_content_lines.append(log_line)

        if current_detailed_text.strip() == '':
            log_line = "Detailed Text:  [Not provided, will use default in Step 1]"
            print(log_line)
            log_content_lines.append(log_line)
        else:
            log_content_lines.append(f"Detailed Text:  {current_detailed_text}")
            dt_preview = current_detailed_text[:150] + "..." if len(current_detailed_text) > 150 else current_detailed_text
            print(f"Detailed Text:  {dt_preview}")
  
        print("="*50 + "\n")
        log_content_lines.append("="*50 + "\n")

        try:
            with open(log_file_path, 'w') as log_f:
                log_f.write("\n".join(log_content_lines))
            print(f"--- Log file saved to: {log_file_path} ---\n")
        except Exception as e:
           print(f"--- WARNING: Could not write log file. Error: {e} ---\n") 

        BLENDER_PATH = "/data/shenkeming/blender-3.6.23-linux-x64/blender" 
        base_motionlcm_dir = "./Motionlcm" 
        MOTIONLCM_RENDER_SCRIPT_PATH = os.path.join(base_motionlcm_dir, "render.py") 

        if not os.path.exists(BLENDER_PATH):
             print("="*50); print(f"Error: Blender executable was not found."); print(f"Expected path: {BLENDER_PATH}"); print("="*50); sys.exit(1)
        if not os.path.exists(MOTIONLCM_RENDER_SCRIPT_PATH):
             print("="*50); print(f"Error: render.py script was not found."); print(f"Expected path: {os.path.abspath(MOTIONLCM_RENDER_SCRIPT_PATH)}"); print("="*50); sys.exit(1) 

        npy_path, total_samples_generated = step1_generate_skeleton(args, base_output_dir, n_frames, current_prompt, current_detailed_text) 
        
        print(f"\n=== Starting to process {total_samples_generated} samples for this ID ===")

        try:
            results_data = np.load(npy_path, allow_pickle=True).item()
            all_motions = results_data['motion']
            all_texts = results_data['text']
            all_lengths = results_data['lengths'] 
        except Exception as e:
            print(f"Error: failed to load {npy_path}. Error: {e}")
            continue

        for i in range(total_samples_generated):
            sample_name = f"sample_{i:02d}" 
            print(f"\n--- Processing Sample {i+1}/{total_samples_generated}: {sample_name} ---") 
            sample_output_dir = os.path.join(base_output_dir, sample_name)
            if not os.path.exists(sample_output_dir): os.makedirs(sample_output_dir) 

            use_cuda = torch.cuda.is_available() and args.device != -1
            device = args.device if use_cuda else 'cpu' 

            sample_motion = all_motions[i] 
            sample_hint = sample_motion[:, 0, :] 

            mesh_pkl_path = step2_convert_to_smpl_sequence(
                npy_path, i, 0, sample_output_dir, sample_name, 
                device, use_cuda, sample_hint 
            ) 

            if mesh_pkl_path is None or not os.path.exists(mesh_pkl_path): 
                print(f"  [ERROR] Mesh file {mesh_pkl_path} was not created by local Step 2.")
                continue 
            
            print(f"  Mesh file created: [{mesh_pkl_path}]")

            print(f"  Running MotionLCM render.py with Blender...(Mode: video)") 
            render_command = (
                f"{BLENDER_PATH} --background --python {MOTIONLCM_RENDER_SCRIPT_PATH} -- "
                f"--pkl {mesh_pkl_path} --mode video --fps 20"
            ) 
            os.system(render_command) 
            output_video_path = mesh_pkl_path.replace(".pkl", ".mp4")
            if os.path.exists(output_video_path):
                print(f"  Successfully rendered video to [{output_video_path}]")
            else:
                print(f"  Warning: Rendered video not found at [{output_video_path}]") 
            
            print(f"  Running MotionLCM render.py with Blender... (Mode: sequence)") 
            render_command_seq = (
                f"{BLENDER_PATH} --background --python {MOTIONLCM_RENDER_SCRIPT_PATH} -- "
                f"--pkl {mesh_pkl_path} --mode sequence --num {args.num_sequence_frames}"
            ) 
            os.system(render_command_seq) 
            output_seq_path = mesh_pkl_path.replace(".pkl", ".png")
            if os.path.exists(output_seq_path):
                print(f"  Successfully rendered sequence to [{output_seq_path}]")
            else:
                print(f"  Warning: Rendered sequence not found at [{output_seq_path}]") 
    
    print(f"\n[SUCCESS] All {num_test_ids_to_run} test runs finished.") 


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        if 'NoSuchDisplayException' in str(e):
             print("\n======================= HINT =======================")
             print("Caught a 'NoSuchDisplayException'. This usually means you are on a headless server.") 
             print("\n1. Using EGL (Recommended):")
             print("   export PYOPENGL_PLATFORM=egl")
             print("   python motionlcm_render.py ...")
             print("\n2. Using a virtual display (Xvfb):")
             print("   xvfb-run python motionlcm_render.py ...")
             print("====================================================") 
        else:
            raise e