import torch

def compare_model_weights(file1, file2, tol=1e-8):
    checkpoint1 = torch.load(file1, map_location="cpu")
    checkpoint2 = torch.load(file2, map_location="cpu")
    
    state_dict1 = checkpoint1['model_state_dict'] if 'model_state_dict' in checkpoint1 else checkpoint1
    state_dict2 = checkpoint2['model_state_dict'] if 'model_state_dict' in checkpoint2 else checkpoint2
    
    all_close = True
    
    total_diff_squared_sum = 0.0
    total_params = 0
    max_diff = 0.0
    max_diff_layer = ""
    all_diffs = []
    
    print("Per-layer difference details:")
    print("=" * 50)
    
    for key in state_dict1:
        if key in state_dict2:
            diff = torch.norm(state_dict1[key] - state_dict2[key]).item()
            num_params = state_dict1[key].numel()
            
            all_diffs.append((key, diff, num_params))
            total_diff_squared_sum += diff ** 2
            total_params += num_params
            
            if diff > max_diff:
                max_diff = diff
                max_diff_layer = key
            
            if diff > tol:
                print(f"Layer {key} differs, L2 difference: {diff:.6f}")
                all_close = False
            else:
                print(f"Layer {key} is close (L2 difference: {diff:.6f})")
        else:
            print(f"Key {key} is missing in the second model.")
            all_close = False
    
    for key in state_dict2:
        if key not in state_dict1:
            print(f"Key {key} is missing in the first model.")
            all_close = False
    
    global_l2_diff = total_diff_squared_sum ** 0.5
    avg_param_diff = global_l2_diff / total_params if total_params > 0 else 0
    
    print("\nOverall difference statistics:")
    print("=" * 50)
    print(f"Global model L2 difference: {global_l2_diff:.6f}")
    print(f"Average per-parameter difference: {avg_param_diff:.8f}")
    print(f"Layer with maximum difference: {max_diff_layer} (difference value: {max_diff:.6f})")
    print(f"Total parameter count: {total_params:,}")
    
    if all_diffs:
        print("\nTop 5 layers with the largest differences:")
        all_diffs.sort(key=lambda x: x[1], reverse=True)
        for i, (key, diff, params) in enumerate(all_diffs[:5]):
            print(f"{i+1}. {key}: difference {diff:.6f}, parameter count {params:,}")
    
    if all_close:
        print("\nWeights match for all layers.")
    else:
        print("\nSome layer weights do not match.")
    
    return global_l2_diff, all_close

file1 = "./save/0401_FineXtrol/model000475000.pt"
file2 = "./save/0401_FineXtrol/model000575000.pt"

total_diff, is_close = compare_model_weights(file1, file2)
print(f"\nTotal model difference: {total_diff:.6f}")