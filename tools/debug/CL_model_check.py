import torch

checkpoint_path = './detailed_text_encoder/T5_base_sequence_MLP_0402.pt'

checkpoint = torch.load(checkpoint_path, map_location='cpu')

if 'mlp_state_dict' in checkpoint:
    print(f"Successfully loaded 'mlp_state_dict'. Parameter keys are listed below:")
    mlp_keys = checkpoint['mlp_state_dict'].keys()

    for key in mlp_keys:
        print(key)

    if not mlp_keys:
        print("\nAnalysis: 'mlp_state_dict' is empty.")
    else:
        print(f"\nAnalysis: 'mlp_state_dict' contains {len(mlp_keys)} parameter tensors.")

else:
    print("Error: 'mlp_state_dict' was not found in the .pt file.")