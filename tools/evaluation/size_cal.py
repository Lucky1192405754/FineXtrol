import torch

def count_parameters(state_dict):
    """Count the total number of parameters in a state dictionary."""
    total_params = sum(param.numel() for param in state_dict.values())
    return total_params

def main():
    # Specify the path to your trained model checkpoint (.pt file)
    # checkpoint_path = './save/0201_T5_pretrained/model000875000.pt'
    checkpoint_path = './save/model000475000.pt'
    
    # Load the checkpoint; map to CPU for compatibility if GPU is unavailable
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    # Check if the checkpoint contains a nested state dictionary
    if 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
        print("Model state dict found. Counting parameters for the main model...")
        total_params = count_parameters(model_state)
    else:
        # Otherwise assume the checkpoint itself is a state dict
        print("No 'model_state_dict' key found. Counting parameters in the checkpoint directly...")
        total_params = count_parameters(checkpoint)
    
    print(f"Total number of parameters: {total_params}")

if __name__ == "__main__":
    main()
