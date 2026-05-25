import torch
from data_loaders.humanml.networks.modules import *
from torch.utils.data import Dataset
from tqdm import tqdm
from utils import dist_util
import sys
import time

total_inference_time = 0
num_inferences = 0

class CompMDMGeneratedDataset(Dataset):

    def __init__(self, model, diffusion, dataloader, mm_num_samples, mm_num_repeats, max_motion_length, num_samples_limit, scale=1.):
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        # print("mm_num_samples:", mm_num_samples) 
        assert mm_num_samples < len(dataloader.dataset)
        use_ddim = False  # FIXME - hardcoded
        clip_denoised = False  # FIXME - hardcoded
        self.max_motion_length = max_motion_length
        sample_fn = (
            diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
        )
        
        real_num_batches = len(dataloader)
        if num_samples_limit is not None:
            real_num_batches = num_samples_limit // dataloader.batch_size + 1
        # print('real_num_batches', real_num_batches) # real_num_batches 32

        generated_motion = []
        mm_generated_motions = []
        # print("mm_num_samples:", mm_num_samples) 
        # sys.exit()
        if mm_num_samples > 0:
            mm_idxs = np.random.choice(real_num_batches, mm_num_samples // dataloader.batch_size +1, replace=False)
            mm_idxs = np.sort(mm_idxs)
        else:
            mm_idxs = []
        # print('mm_idxs', mm_idxs) 

        model.eval()
        count = 1

        with torch.no_grad():
            for i, (motion, model_kwargs) in tqdm(enumerate(dataloader), miniters=0.1, dynamic_ncols=True):
                # print("Begin test...")
                for k, v in model_kwargs['y'].items():
                    # print("k:", k)
                    # print("v:", v)
                    if torch.is_tensor(v):
                        model_kwargs['y'][k] = v.to(dist_util.dev())

                if num_samples_limit is not None and len(generated_motion) >= num_samples_limit:
                    break
                
                tokens = [t.split('_') for t in model_kwargs['y']['tokens']]

                # add CFG scale to batch
                if scale != 1.:
                    model_kwargs['y']['scale'] = torch.ones(motion.shape[0],
                                                            device=dist_util.dev()) * scale

                mm_num_now = len(mm_generated_motions) // dataloader.batch_size # dataloader.batch_size
                is_mm = i in mm_idxs
                # print("is_mm:", is_mm) # is_mm: False
                
                repeat_times = mm_num_repeats if is_mm else 1
                # print("repeat_times:", repeat_times) # repeat_times: 1
                
                mm_motions = []
                for t in range(repeat_times):
                    start_time = time.time()
                    sample = sample_fn(
                        model,
                        motion.shape,
                        clip_denoised=clip_denoised,
                        model_kwargs=model_kwargs,
                        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                        init_image=None,
                        progress=False,
                        dump_steps=None,
                        noise=None,
                        const_noise=False,
                        # when experimenting guidance_scale we want to nutrileze the effect of noise on generation
                    )
                    torch.cuda.synchronize()
                    end_time = time.time()
                    # total_inference_time += (end_time - start_time)
                    # print("infer time", total_inference_time)
                    
                    if t == 0:
                        sub_dicts = [{'motion': sample[bs_i].squeeze().permute(1,0).cpu().numpy(),
                                    'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
                                    'caption': model_kwargs['y']['text'][bs_i],
                                    # 'hint': model_kwargs['y']['hint'][bs_i].cpu().numpy() if 'hint' in model_kwargs['y'] else None,
                                    'detailed_text': model_kwargs['y']['detailed_text'][bs_i] if 'detailed_text' in model_kwargs['y'] else None,
                                    'tokens': tokens[bs_i],
                                    'cap_len': len(tokens[bs_i]),
                                    } for bs_i in range(dataloader.batch_size)] # 32 iteration dataloader.batch_size
                        generated_motion += sub_dicts
                        # print("generated_motion:", generated_motion)
                        # print(generated_motion.shape)
                        # print("detailed text:", generated_motion['detailed_text'])

                    if is_mm:
                        mm_motions += [{'motion': sample[bs_i].squeeze().permute(1, 0).cpu().numpy(),
                                        'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
                                        } for bs_i in range(dataloader.batch_size)] # dataloader.batch_size

                if is_mm:
                    mm_generated_motions += [{
                                    'caption': model_kwargs['y']['text'][bs_i],
                                    'tokens': tokens[bs_i],
                                    'cap_len': len(tokens[bs_i]),
                                    'mm_motions': mm_motions[bs_i::dataloader.batch_size],  # collect all 10 repeats from the (32*10) generated motions
                                    } for bs_i in range(dataloader.batch_size)] # dataloader.batch_size

        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions
        self.w_vectorizer = dataloader.dataset.w_vectorizer


    def __len__(self):
        return len(self.generated_motion)


    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length, caption, tokens, hint = data['motion'], data['length'], data['caption'], data['tokens'], data['detailed_text']
        # print("hint:", hint)
        sent_len = data['cap_len']

        if self.dataset.mode == 'eval':
            normed_motion = motion
            denormed_motion = self.dataset.t2m_dataset.inv_transform(normed_motion)
            renormed_motion = (denormed_motion - self.dataset.mean_for_eval) / self.dataset.std_for_eval  # according to T2M norms
            motion = renormed_motion
            # This step is needed because T2M evaluators expect their norm convention

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # print("hint:", hint)
        # print("=====eval code checked here!=====")
        # sys.exit()
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), hint