import torch
import sys
from transformers import T5Tokenizer, T5EncoderModel

MODEL_NAME_HF = "google-t5/t5-base"
MODEL_PATH_LOCAL = "./t5-base-local"
SAMPLE_TEXT = "Verify this text tokenization."

print("--- T5 local model verification script ---")

try:
    print("\n[Step 1/4] Loading tokenizers...")
    tokenizer_hf = T5Tokenizer.from_pretrained(MODEL_NAME_HF, legacy=False)
    tokenizer_local = T5Tokenizer.from_pretrained(MODEL_PATH_LOCAL, legacy=False)
    print("Tokenizers loaded.")

    print("\n[Step 2/4] Verifying tokenizers...")
    tokens_hf = tokenizer_hf(SAMPLE_TEXT, return_tensors="pt").input_ids
    tokens_local = tokenizer_local(SAMPLE_TEXT, return_tensors="pt").input_ids

    if torch.equal(tokens_hf, tokens_local):
        print("  [SUCCESS] Tokenizer verification passed. Output IDs are identical.")
    else:
        print("  [FAILURE] Tokenizer verification failed.")
        print(f"    HuggingFace Output: {tokens_hf}")
        print(f"    Local output:     {tokens_local}")
        sys.exit(1)

    print("\n[Step 3/4] Loading models; this may take some time...")
    model_hf = T5EncoderModel.from_pretrained(MODEL_NAME_HF)
    model_hf.eval()
    model_local = T5EncoderModel.from_pretrained(MODEL_PATH_LOCAL)
    model_local.eval()
    print("Models loaded.")

    print("\n[Step 4/4] Comparing model parameters one by one...")
    state_dict_hf = model_hf.state_dict()
    state_dict_local = model_local.state_dict()

    if len(state_dict_hf) != len(state_dict_local):
        print(f"  [FAILURE] Parameter counts do not match. (HF: {len(state_dict_hf)}, Local: {len(state_dict_local)})")
        sys.exit(1)

    all_params_match = True
    for (key_hf, param_hf), (key_local, param_local) in zip(state_dict_hf.items(), state_dict_local.items()):
        
        if key_hf != key_local:
            print(f"  [FAILURE] Parameter keys do not match: {key_hf} vs {key_local}")
            all_params_match = False
            break
        
        if not torch.allclose(param_hf, param_local):
            print(f"  [FAILURE] Parameter values do not match: {key_hf}")
            all_params_match = False
            break

    if all_params_match:
        print("  [SUCCESS] Model verification passed. All model weights are identical.")
    else:
        print("  [FAILURE] Model verification failed.")
        sys.exit(1)
        
    print("\n--- [ Verification passed ] ---")
    print("You can safely replace the model path with './t5-base-local' in CMDM.txt.")

except Exception as e:
    print(f"\n[ An error occurred during verification ]")
    print(f"Error message: {e}")
    print("Please make sure your network connection is stable while running this script and './t5-base-local' is correct.")