# This code is modified based on https://github.com/GuyTevet/motion-diffusion-model
import numpy as np
import torch
import torch.nn as nn
import clip
from model.rotation2xyz import Rotation2xyz
from .transformer import *
import sys
from transformers import T5Tokenizer, T5EncoderModel

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class CMDM(torch.nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                 latent_dim=512, # input from command!!!
                 ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
                 arch='trans_enc', emb_trans_dec=False, clip_version=None, *args, **kargs):
        super().__init__() 

        print("initialize T5 base model...")
        # HF T5-base
        # self.tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base", legacy=False)  # Adjust this to your specific model
        # self.t5EncoderModel = T5EncoderModel.from_pretrained("google-t5/t5-base")

        # Local T5-base
        local_t5_path = "./t5-base-local" 
        
        print(f"Loading T5 Tokenizer from local path: {local_t5_path}")
        self.tokenizer = T5Tokenizer.from_pretrained(local_t5_path, legacy=False)
        
        print(f"Loading T5 EncoderModel from local path: {local_t5_path}")
        self.t5EncoderModel = T5EncoderModel.from_pretrained(local_t5_path)

        self.t5EncoderModel.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        
        t5_ckpt_dir = './detailed_text_encoder/T5_base_sequence_MLP_0402.pt'
        # t5_ckpt_dir = './detailed_text_encoder/T5_base_sentence_level_0328_3MLPs.pt'
        # t5_ckpt_dir = './detailed_text_encoder/T5_snippet_level_12_01.pt'


        print(f"loading pre-trained T5 model from {t5_ckpt_dir}...")
        checkpoint = torch.load(t5_ckpt_dir)

        checkpoint_dict = {k.replace('module.', ''): v for k, v in checkpoint["model_state_dict"].items()}

        missing_keys, unexpected_keys = self.t5EncoderModel.load_state_dict(checkpoint_dict, strict=False)

        print("missing_keys:", missing_keys)
        print("unexpected_keys:", unexpected_keys)
        print("Model loaded successfully")

        # print("\nVerifying the loaded T5 model against the base model...")
        # with torch.no_grad():
            # base_t5_model = T5EncoderModel.from_pretrained("google-t5/t5-base")
            # base_t5_model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            # base_t5_model.eval()

        #     all_params_match = True
            
        #     for (name1, param1), (name2, param2) in zip(self.t5EncoderModel.named_parameters(), base_t5_model.named_parameters()):
        #         if name1 != name2:
        #             print(f"Parameter name mismatch: {name1} vs {name2}")
        #             all_params_match = False
        #             break
                
        #         if not torch.allclose(param1, param2):
        #             print(f"Found difference in parameter: {name1}. Verification successful!")
        #             all_params_match = False
            
        #     if all_params_match:
        #         print("Verification FAILED: The loaded model's parameters are identical to the original T5-base model.")
        #     else:
        #         print("Verification PASSED: The loaded model's parameters differ from the T5-base, indicating the fine-tuned model was loaded correctly.")
            
        #     del base_t5_model
        # print("-" * 50)


        # print("initialize T5 base model...")

        # print("T5-base model loaded successfully")

        self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_actions = num_actions
        self.data_rep = data_rep
        self.dataset = dataset

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim
        print("1-latent_dim: ",self.latent_dim)

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        self.clip_dim = clip_dim
        self.action_emb = kargs.get('action_emb', None)

        self.input_feats = self.njoints * self.nfeats

        self.normalize_output = kargs.get('normalize_encoder_output', False)

        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.arch = arch
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0
        self.emb_trans_dec = emb_trans_dec

        self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)
        # --- MDM ---
        self.input_process = InputProcess(self.data_rep, self.input_feats+self.gru_emb_dim, self.latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        print("TRANS_ENC init")
        seqTransEncoderLayer = TransformerEncoderLayer(d_model=self.latent_dim,
                                                        nhead=self.num_heads,
                                                        dim_feedforward=self.ff_size,
                                                        dropout=self.dropout,
                                                        activation=self.activation)

        self.seqTransEncoder = TransformerEncoder(seqTransEncoderLayer,
                                                num_layers=self.num_layers)

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        if self.cond_mode != 'no_cond':
            if 'text' in self.cond_mode:
                self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
                print("Current CLIP dims:", self.clip_dim, self.latent_dim)
                self.t5_embed_text = nn.Linear(768, 512) # hard-code for T5 detailed text
                print("Current T5 dims:", 768, 512)
                print('EMBED TEXT')
                print('Loading CLIP & T5...')
                self.clip_version = clip_version
                self.clip_model = self.load_and_freeze_clip(clip_version)

        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints,
                                            self.nfeats)
        # ------
        # --- CMDM ---
        # input 263 or 6 * 3 or 3
        n_joints = 22 if njoints == 263 else 21
        self.input_hint_block = HintBlock(self.data_rep, n_joints * 3, self.latent_dim) # TODO

        self.c_input_process = InputProcess(self.data_rep, self.input_feats+self.gru_emb_dim, self.latent_dim)

        self.c_sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        print("TRANS_ENC init")
        seqTransEncoderLayer = TransformerEncoderLayer(d_model=self.latent_dim,
                                                        nhead=self.num_heads,
                                                        dim_feedforward=self.ff_size,
                                                        dropout=self.dropout,
                                                        activation=self.activation)
        self.c_seqTransEncoder = TransformerEncoder(seqTransEncoderLayer,
                                                    num_layers=self.num_layers,
                                                    return_intermediate=True)

        self.zero_convs = zero_module(nn.ModuleList([nn.Linear(self.latent_dim, self.latent_dim) for _ in range(self.num_layers)]))
        
        self.c_embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        if self.cond_mode != 'no_cond':
            if 'text' in self.cond_mode:
                self.c_embed_text = nn.Linear(self.clip_dim, self.latent_dim)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]
    
    def flatten_text_input(self, text_list):
        """ Flatten list of list texts if necessary. """
        return [c for sub in text_list for c in sub] if isinstance(text_list[0], list) else text_list

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    # def mask_cond(self, cond, force_mask=False):
    #     bs, d = cond.shape
    #     if force_mask:
    #         return torch.zeros_like(cond)
    #     elif self.training and self.cond_mask_prob > 0.:
    #         mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
    #         return cond * (1. - mask)
    #     else:
    #         return cond
    

    def mask_cond(self, cond, force_mask=False):
        if cond.dim() == 3:
            cond = cond.squeeze(0)
        assert cond.dim() == 2, f"Expected 2D tensor, got shape {cond.shape}"

        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)
            return cond * (1. - mask)
        else:
            return cond

    
    
    # def encode_text(self, raw_text):
    #     """
    #     """
    #     device = next(self.parameters()).device
    #     if isinstance(raw_text[0], (list, tuple)):
    #         bs, T = len(raw_text), len(raw_text[0])
    #         # tokenize + encode
    #         texts = clip.tokenize(flat, truncate=True).to(device)  # [bs*T, L]
    #         feats = self.clip_model.encode_text(texts).float()     # [bs*T, clip_dim]
    #         return feats.view(bs, T, -1)
    #     else:
    #         texts = clip.tokenize(raw_text, truncate=True).to(device)  # [bs, L]
    #         return self.clip_model.encode_text(texts).float()         # [bs, clip_dim]
    
    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        # max_text_len = None if self.dataset in ['humanml', 'kit'] else 20  # Specific hardcoding for humanml dataset
        max_text_len = 20 if self.dataset in ['humanml', 'kit'] else None  # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            # context_length = default_context_length - 1 # 76
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            # print('texts', texts.shape)
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float()
    
    def T5_encode_text(self, raw_text):
        """
        Encode the detailed text using a fine-tuned T5 model.

        raw_text: list of strings (batch_size length)
        Returns: Encoded text representations (torch.Tensor)
        """
        inputs = self.tokenizer(raw_text, padding=True, truncation=True, return_tensors="pt", max_length=512)
        device = next(self.parameters()).device

        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            encoded_outputs = self.t5EncoderModel(**inputs)
            encoded_text = encoded_outputs.last_hidden_state
        
        encoded_text = encoded_text.mean(dim=1) # [batch_size, hidden_dim]

        return encoded_text.float()
    
    # def T5_encode_text(self, raw_text):
    #     """
    #     """
    #     if isinstance(raw_text[0], (list, tuple)):
    #         bs = len(raw_text)
    #         T = len(raw_text[0])
    #         flat = [s for sub in raw_text for s in sub]
    #         inputs = self.tokenizer(flat, padding=True, truncation=True,
    #                                 return_tensors="pt", max_length=512)
    #         device = next(self.parameters()).device
    #         inputs = {k: v.to(device) for k, v in inputs.items()}
    #         with torch.no_grad():
    #             out = self.t5EncoderModel(**inputs).last_hidden_state  # [bs*T, L, H]
    #         # pool to [bs*T, H]
    #         out = out.mean(dim=1)
    #         out = out.view(bs, T, -1)
    #         return out.float()
    #     else:
    #         inputs = self.tokenizer(raw_text, padding=True, truncation=True,
    #                                 return_tensors="pt", max_length=512)
    #         device = next(self.parameters()).device
    #         inputs = {k: v.to(device) for k, v in inputs.items()}
    #         with torch.no_grad():
    #             out = self.t5EncoderModel(**inputs).last_hidden_state  # [bs, L, H]
    #         # pool
    #         return out.mean(dim=1).float()  # [bs, H]
        
    # def cmdm_forward(self, x, timesteps, y=None, weight=1.0):
    #     """
    #     Realism Guidance
    #     x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
    #     timesteps: [batch_size] (int)
    #     """

    #     bs, nj, feats, frames = x.shape # torch.Size([64, 263, 1, 196])
    #     T = len(y['detailed_text'][0])  # e.g. 4
    #     x = x.unsqueeze(1).repeat(1, T, 1, 1, 1) # [bs, T, nj, feats, frames] torch.Size([64, 4, 263, 1, 196])
    #     x = x.view(bs*T, nj, feats, frames)      # [bs*T, nj, feats, frames] torch.Size([256, 263, 1, 196])

    #     timesteps = timesteps.unsqueeze(1).repeat(1, T)   # [bs, T] torch.Size([64, 4])
    #     timesteps = timesteps.view(bs*T)                 # [bs*T] torch.Size([256])

    #     emb = self.embed_timestep(timesteps)     # torch.Size([1, 256, 512])
    #     # emb = emb.unsqueeze(0) # torch.Size([1, 1, 64, 512])
    #     # sys.exit()


    #     # print(y['text'])

    #     if 'detailed_text' in y.keys():
    #         # —— FineGrained —— #
    #         # print("Raw detailed_text:", y['detailed_text'])
    #         guided_hint = self.T5_encode_text(flat_detail) # using T5-base or pretrained model torch.Size([256, 768])
    #         # guided_hint = self.T5_encode_text(y['detailed_text']) # using T5-base or pretrained model
    #         guided_hint = self.t5_embed_text(guided_hint) # torch.Size([256, 512])
    #         # guided_hint = guided_hint.permute(1, 0, 2)
    #         guided_hint = guided_hint.unsqueeze(0) # torch.Size([1, 256, 512])

    #         # —— CoarseGrained —— #
    #         # enc_text = self.encode_text(y['text'])
    #         enc_text = self.encode_text(flat_coarse)
            
            
    #         # enc_proj = self.embed_text(enc_text)          # [bs*T, D] torch.Size([256, 512])
    #         # enc_proj = enc_proj.permute(1, 0, 2)        # [T, bs, D]
    #         # enc_proj = enc_proj.unsqueeze(0)              # torch.Size([1, 256, 512])
    #         force_mask = y.get('uncond', False)
    #         masked_enc = self.mask_cond(enc_text, force_mask=force_mask)  # [256, 512]
    #         masked_enc_proj = self.embed_text(masked_enc)                 # [256, 512]

    #         # emb torch.Size([1, 64, 512])
    #         # enc_proj torch.Size([1, 256, 512])
    #         emb = emb + masked_enc_proj.unsqueeze(0)      # [1, 256, 512]
    #         x = self.c_input_process(x)
    #         x += guided_hint # torch.Size([196, 256, 512])
    #         # sys.exit()

    #     else:
    #         seq_mask = y['hint'].sum(-1) != 0
    #         guided_hint = self.input_hint_block(y['hint'].float())  # [bs, d]
    #         force_mask = y.get('uncond', False)
    #         if 'text' in self.cond_mode:
    #             enc_text = self.encode_text(y['text'])
    #             emb += self.c_embed_text(self.mask_cond(enc_text, force_mask=force_mask))
    #         x = self.c_input_process(x)
    #         x += guided_hint * seq_mask.permute(1, 0).unsqueeze(-1)
    
    #     # adding the timestep embed
    #     xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
    #     xseq = self.c_sequence_pos_encoder(xseq)  # [seqlen+1, bs, d] 
    #     output = self.c_seqTransEncoder(xseq)  # [seqlen+1, bs, d]

    #     control = []
    #     for i, module in enumerate(self.zero_convs):
    #         control.append(module(output[i]))
    #     control = torch.stack(control)

    #     control = control * weight
    #     return control
    
    def cmdm_forward(self, x, timesteps, y=None, weight=1.0):
        """
        Realism Guidance
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """

        # if y is not None:
        #     print("="*50)
            
        #     if 'text' in y:
        #     else:

        #     if 'detailed_text' in y:
        #     else:
        #     print("="*50)
        
        # sys.exit()

        emb = self.embed_timestep(timesteps)  # [1, bs, d]
        
        if 'detailed_text' in y.keys():
            # print("Raw detailed_text:", y['detailed_text'])
            guided_hint = self.T5_encode_text(y['detailed_text']) # using T5-base or pretrained model
            # print(f"[DEBUG_0]The shape of current tensor is: {guided_hint.shape}")
            # print("Encoded detailed_text shape:", guided_hint.shape)
            # print("Encoded detailed_text sample (first 5 values):", guided_hint[0][:5].detach().cpu().numpy())

            guided_hint = self.t5_embed_text(guided_hint)
            # print(f"[DEBUG_1]The shape of current tensor is: {guided_hint.shape}")
            # print("T5 projected detailed_text shape:", guided_hint.shape)
            # print("T5 projected detailed_text sample (first 5 values):", guided_hint[0][:5].detach().cpu().numpy())
            # sys.exit()
            force_mask = y.get('uncond', False)
            if 'text' in self.cond_mode:
                enc_text = self.encode_text(y['text'])
                # print(f"[DEBUG_2]The shape of current tensor is: {enc_text.shape}")
                emb += self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))
                # print(f"[DEBUG_3]The shape of current tensor is: {emb.shape}")

            # print(f"[DEBUG_4]The shape of current tensor is: {x.shape}")
            x = self.c_input_process(x)
            # print(f"[DEBUG_5]The shape of current tensor is: {x.shape}")
            x += guided_hint
            # print(f"[DEBUG_6]The shape of current tensor is: {x.shape}")

        else:
            seq_mask = y['hint'].sum(-1) != 0
            guided_hint = self.input_hint_block(y['hint'].float())  # [bs, d]
            force_mask = y.get('uncond', False)
            if 'text' in self.cond_mode:
                enc_text = self.encode_text(y['text'])
                emb += self.c_embed_text(self.mask_cond(enc_text, force_mask=force_mask))
            x = self.c_input_process(x)
            x += guided_hint * seq_mask.permute(1, 0).unsqueeze(-1)
    
        # adding the timestep embed
        xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
        xseq = self.c_sequence_pos_encoder(xseq)  # [seqlen+1, bs, d] 
        output = self.c_seqTransEncoder(xseq)  # [seqlen+1, bs, d]

        control = []
        for i, module in enumerate(self.zero_convs):
            control.append(module(output[i]))
        control = torch.stack(control)

        control = control * weight
        return control

    
    # def mdm_forward(self, x, timesteps, y=None, control=None):
    #     """
    #     x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
    #     timesteps: [batch_size] (int)
    #     """
    #     bs = x.shape[0]
    #     T = 4
    #     timesteps = timesteps.unsqueeze(1).repeat(1, T)   # [bs, T] torch.Size([64, 4])
    #     timesteps = timesteps.view(bs*T)                 # [bs*T] torch.Size([256])

    #     emb = self.embed_timestep(timesteps)  # [1, bs, d]

    #     force_mask = y.get('uncond', False)

    #     if 'text' in self.cond_mode:
    #         flat_text = self.flatten_text_input(y['text'])           # flatten to [bs*T]
    #         enc_text = self.encode_text(flat_text)              # [bs*T, 512]
    #         masked_enc = self.mask_cond(enc_text, force_mask)   # [bs*T, 512]
    #         masked_enc_proj = self.embed_text(masked_enc)       # [bs*T, 512]
    #         emb = emb + masked_enc_proj.unsqueeze(0)            # [1, bs*T, 512]

    #     x = self.input_process(x)                               # [seqlen, bs*T, d]

    #     # Add timestep embedding
    #     xseq = torch.cat((emb, x), axis=0)                      # [seqlen+1, bs*T, d]
    #     xseq = self.sequence_pos_encoder(xseq)                  # [seqlen+1, bs*T, d]
    #     output = self.seqTransEncoder(xseq, control=control)[1:]  # skip timestep token

    #     output = self.output_process(output)                    # [bs*T, njoints, nfeats, nframes]
    #     return output
    
    # Original mdm_forward function
    def mdm_forward(self, x, timesteps, y=None, control=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        emb = self.embed_timestep(timesteps)  # [1, bs, d]

        force_mask = y.get('uncond', False)
        if 'text' in self.cond_mode:
            enc_text = self.encode_text(y['text'])
            emb += self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))

        x = self.input_process(x)

        # adding the timestep embed
        xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
        xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        output = self.seqTransEncoder(xseq, control=control)[1:]  

        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        return output
    
    
    # original forward function
    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        if 'detailed_text' in y.keys():
            # detailed_text = y['detailed_text']
            # print("Detailed text input in model:", detailed_text)
            # sys.exit()
            control = self.cmdm_forward(x, timesteps, y)

        else:
            n_joints = 22 if self.njoints == 263 else 21
            y_ = {'hint': torch.zeros((x.shape[0], x.shape[-1], n_joints * 3), device=x.device)}
            y_.update(y)
            control = self.cmdm_forward(x, timesteps, y_)

        output = self.mdm_forward(x, timesteps, y, control)
        return output
    
    def _apply(self, fn):
        super()._apply(fn)
        self.rot2xyz.smpl_model._apply(fn)


    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        self.rot2xyz.smpl_model.train(*args, **kwargs)

#===============================================================================================================
# class CMDM(torch.nn.Module):
#     def __init__(self, modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
#                  latent_dim=512, # input from command!!!
#                  ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
#                  ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
#                  arch='trans_enc', emb_trans_dec=False, clip_version=None, *args, **kargs):
#         super().__init__() 

#         print("initialize T5 base model...")
#         self.tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base", legacy=False)  # Adjust this to your specific model
#         self.t5EncoderModel = T5EncoderModel.from_pretrained("google-t5/t5-base")
#         self.t5EncoderModel.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        
#         t5_ckpt_dir = './detailed_text_encoder/T5_base_sequence_MLP_0402.pt'
#         print(f"loading pre-trained T5 model from {t5_ckpt_dir}...")
#         checkpoint = torch.load(t5_ckpt_dir)

#         checkpoint_dict = {k.replace('module.', ''): v for k, v in checkpoint["model_state_dict"].items()}

#         missing_keys, unexpected_keys = self.t5EncoderModel.load_state_dict(checkpoint_dict, strict=False)

#         print("missing_keys:", missing_keys)
#         print("unexpected_keys:", unexpected_keys)
#         print("Model loaded successfully")

#         # print("initialize T5 base model...")

#         # print("T5-base model loaded successfully")

#         self.legacy = legacy
#         self.modeltype = modeltype
#         self.njoints = njoints
#         self.nfeats = nfeats
#         self.num_actions = num_actions
#         self.data_rep = data_rep
#         self.dataset = dataset

#         self.pose_rep = pose_rep
#         self.glob = glob
#         self.glob_rot = glob_rot
#         self.translation = translation

#         self.latent_dim = latent_dim
#         print("1-latent_dim: ",self.latent_dim)

#         self.ff_size = ff_size
#         self.num_layers = num_layers
#         self.num_heads = num_heads
#         self.dropout = dropout

#         self.ablation = ablation
#         self.activation = activation
#         self.clip_dim = clip_dim
#         self.action_emb = kargs.get('action_emb', None)

#         self.input_feats = self.njoints * self.nfeats

#         self.normalize_output = kargs.get('normalize_encoder_output', False)

#         self.cond_mode = kargs.get('cond_mode', 'no_cond')
#         self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
#         self.arch = arch
#         self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0
#         self.emb_trans_dec = emb_trans_dec

#         self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)
#         # --- MDM ---
#         self.input_process = InputProcess(self.data_rep, self.input_feats+self.gru_emb_dim, self.latent_dim)
#         self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

#         print("TRANS_ENC init")
#         seqTransEncoderLayer = TransformerEncoderLayer(d_model=self.latent_dim,
#                                                         nhead=self.num_heads,
#                                                         dim_feedforward=self.ff_size,
#                                                         dropout=self.dropout,
#                                                         activation=self.activation)

#         self.seqTransEncoder = TransformerEncoder(seqTransEncoderLayer,
#                                                 num_layers=self.num_layers)

#         self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

#         if self.cond_mode != 'no_cond':
#             if 'text' in self.cond_mode:
#                 self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
#                 print("Current CLIP dims:", self.clip_dim, self.latent_dim)
#                 self.t5_embed_text = nn.Linear(768, 512) # hard-code for T5 detailed text
#                 print("Current T5 dims:", 768, 512)
#                 print('EMBED TEXT')
#                 print('Loading CLIP & T5...')
#                 self.clip_version = clip_version
#                 self.clip_model = self.load_and_freeze_clip(clip_version)

#         self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints,
#                                             self.nfeats)
#         # ------
#         # --- CMDM ---
#         # input 263 or 6 * 3 or 3
#         n_joints = 22 if njoints == 263 else 21
#         self.input_hint_block = HintBlock(self.data_rep, n_joints * 3, self.latent_dim) # TODO

#         self.c_input_process = InputProcess(self.data_rep, self.input_feats+self.gru_emb_dim, self.latent_dim)

#         self.c_sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

#         print("TRANS_ENC init")
#         seqTransEncoderLayer = TransformerEncoderLayer(d_model=self.latent_dim,
#                                                         nhead=self.num_heads,
#                                                         dim_feedforward=self.ff_size,
#                                                         dropout=self.dropout,
#                                                         activation=self.activation)
#         self.c_seqTransEncoder = TransformerEncoder(seqTransEncoderLayer,
#                                                     num_layers=self.num_layers,
#                                                     return_intermediate=True)

#         self.zero_convs = zero_module(nn.ModuleList([nn.Linear(self.latent_dim, self.latent_dim) for _ in range(self.num_layers)]))
        
#         self.c_embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

#         if self.cond_mode != 'no_cond':
#             if 'text' in self.cond_mode:
#                 self.c_embed_text = nn.Linear(self.clip_dim, self.latent_dim)

#     def parameters_wo_clip(self):
#         return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

#     def load_and_freeze_clip(self, clip_version):
#         clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
#                                                 jit=False)  # Must set jit=False for training
#         clip.model.convert_weights(
#             clip_model)  # Actually this line is unnecessary since clip by default already on float16

#         # Freeze CLIP weights
#         clip_model.eval()
#         for p in clip_model.parameters():
#             p.requires_grad = False

#         return clip_model

#     def mask_cond(self, cond, force_mask=False):
#         bs, d = cond.shape
#         if force_mask:
#             return torch.zeros_like(cond)
#         elif self.training and self.cond_mask_prob > 0.:
#             mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
#             return cond * (1. - mask)
#         else:
#             return cond

#     def encode_text(self, raw_text):
#         # raw_text - list (batch_size length) of strings with input text prompts
#         device = next(self.parameters()).device
#         # max_text_len = None if self.dataset in ['humanml', 'kit'] else 20  # Specific hardcoding for humanml dataset
#         max_text_len = 20 if self.dataset in ['humanml', 'kit'] else None  # Specific hardcoding for humanml dataset
#         if max_text_len is not None:
#             default_context_length = 77
#             context_length = max_text_len + 2 # start_token + 20 + end_token
#             # context_length = default_context_length - 1 # 76
#             assert context_length < default_context_length
#             texts = clip.tokenize(raw_text, context_length=context_length).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
#             # print('texts', texts.shape)
#             zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
#             texts = torch.cat([texts, zero_pad], dim=1)
#             # print('texts after pad', texts.shape, texts)
#         else:
#             texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
#         return self.clip_model.encode_text(texts).float()
    
#     def T5_encode_text(self, raw_text):
#         """
#         Encode the detailed text using a fine-tuned T5 model.

#         raw_text: list of strings (batch_size length)
#         Returns: Encoded text representations (torch.Tensor)
#         """
#         inputs = self.tokenizer(raw_text, padding=True, truncation=True, return_tensors="pt", max_length=512)
#         device = next(self.parameters()).device

#         inputs = {key: value.to(device) for key, value in inputs.items()}

#         with torch.no_grad():
#             encoded_outputs = self.t5EncoderModel(**inputs)
#             encoded_text = encoded_outputs.last_hidden_state
        
#         encoded_text = encoded_text.mean(dim=1) # [batch_size, hidden_dim]

#         return encoded_text.float()

#     def cmdm_forward(self, x, timesteps, y=None, weight=1.0):
#         """
#         Realism Guidance
#         x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
#         timesteps: [batch_size] (int)
#         """
#         emb = self.embed_timestep(timesteps)  # [1, bs, d]
        
#         if 'detailed_text' in y.keys():
#             # print("Raw detailed_text:", y['detailed_text'])
#             guided_hint = self.T5_encode_text(y['detailed_text']) # using T5-base or pretrained model
#             # print("Encoded detailed_text shape:", guided_hint.shape)
#             # print("Encoded detailed_text sample (first 5 values):", guided_hint[0][:5].detach().cpu().numpy())

#             guided_hint = self.t5_embed_text(guided_hint)
#             # print("T5 projected detailed_text shape:", guided_hint.shape)
#             # print("T5 projected detailed_text sample (first 5 values):", guided_hint[0][:5].detach().cpu().numpy())
#             # sys.exit()
#             force_mask = y.get('uncond', False)
#             if 'text' in self.cond_mode:
#                 enc_text = self.encode_text(y['text'])
#                 emb += self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))
#             x = self.c_input_process(x)
#             x += guided_hint

#         else:
#             seq_mask = y['hint'].sum(-1) != 0
#             guided_hint = self.input_hint_block(y['hint'].float())  # [bs, d]
#             force_mask = y.get('uncond', False)
#             if 'text' in self.cond_mode:
#                 enc_text = self.encode_text(y['text'])
#                 emb += self.c_embed_text(self.mask_cond(enc_text, force_mask=force_mask))
#             x = self.c_input_process(x)
#             x += guided_hint * seq_mask.permute(1, 0).unsqueeze(-1)
    
#         # adding the timestep embed
#         xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
#         xseq = self.c_sequence_pos_encoder(xseq)  # [seqlen+1, bs, d] 
#         output = self.c_seqTransEncoder(xseq)  # [seqlen+1, bs, d]

#         control = []
#         for i, module in enumerate(self.zero_convs):
#             control.append(module(output[i]))
#         control = torch.stack(control)

#         control = control * weight
#         return control
    
#     def mdm_forward(self, x, timesteps, y=None, control=None):
#         """
#         x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
#         timesteps: [batch_size] (int)
#         """
#         emb = self.embed_timestep(timesteps)  # [1, bs, d]

#         force_mask = y.get('uncond', False)
#         if 'text' in self.cond_mode:
#             enc_text = self.encode_text(y['text'])
#             emb += self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))

#         x = self.input_process(x)

#         # adding the timestep embed
#         xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
#         xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
#         output = self.seqTransEncoder(xseq, control=control)[1:]  

#         output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
#         return output

#     def forward(self, x, timesteps, y=None):
#         """
#         x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
#         timesteps: [batch_size] (int)
#         """
#         if 'detailed_text' in y.keys():
#             # detailed_text = y['detailed_text']
#             # print("Detailed text input in model:", detailed_text)
#             control = self.cmdm_forward(x, timesteps, y)

#         else:
#             n_joints = 22 if self.njoints == 263 else 21
#             y_ = {'hint': torch.zeros((x.shape[0], x.shape[-1], n_joints * 3), device=x.device)}
#             y_.update(y)
#             control = self.cmdm_forward(x, timesteps, y_)

#         output = self.mdm_forward(x, timesteps, y, control)
#         return output

#     def _apply(self, fn):
#         super()._apply(fn)
#         self.rot2xyz.smpl_model._apply(fn)


#     def train(self, *args, **kwargs):
#         super().train(*args, **kwargs)
#         self.rot2xyz.smpl_model.train(*args, **kwargs)
# ==============================================================================================================


class HintBlock(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.ModuleList([
            nn.Linear(self.input_feats, self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim),
            zero_module(nn.Linear(self.latent_dim, self.latent_dim))
        ])

    def forward(self, x):
        x = x.permute((1, 0, 2))

        for module in self.poseEmbedding:
            x = module(x)  # [seqlen, bs, d]
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)

class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.data_rep == 'rot_vel':
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)

        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            x = self.poseEmbedding(x)  # [seqlen, bs, d]
            return x
        elif self.data_rep == 'rot_vel':
            first_pose = x[[0]]  # [1, bs, 150]
            first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
            vel = x[1:]  # [seqlen-1, bs, 150]
            vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
            return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        else:
            raise ValueError


class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        if self.data_rep == 'rot_vel':
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            output = self.poseFinal(output)  # [seqlen, bs, 150]
        elif self.data_rep == 'rot_vel':
            first_pose = output[[0]]  # [1, bs, d]
            first_pose = self.poseFinal(first_pose)  # [1, bs, 150]
            vel = output[1:]  # [seqlen-1, bs, d]
            vel = self.velFinal(vel)  # [seqlen-1, bs, 150]
            output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
        else:
            raise ValueError
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output


class EmbedAction(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        return output