import torch
from collections import defaultdict, OrderedDict

def analyze_nested_dict(d, module_params, total_params, parent_key=''):
    """
    Recursively traverse a dictionary to find and analyze all torch.Tensor parameters.
    d: Dictionary currently being traversed.
    module_params: Dictionary that accumulates parameters by module name.
    total_params: List accumulating the total parameter count, used to pass by reference between functions.
    parent_key: Used to build the full parameter name.
    """
    for k, v in d.items():
        full_key = f"{parent_key}.{k}" if parent_key else k
        
        if isinstance(v, (dict, OrderedDict)):
            analyze_nested_dict(v, module_params, total_params, parent_key=full_key)
        elif isinstance(v, torch.Tensor):
            num_params = v.numel()
            total_params[0] += num_params
            
            module_name = full_key.split('.')[0]
            module_params[module_name] += num_params

model_path = 'save/1012_test/model000875235.pt'
print(f"--- Analyzing file: {model_path} ---")

try:
    data = torch.load(model_path, map_location='cpu')
    print(f"--- Loaded data type: {type(data)} ---")
except Exception as e:
    print(f"Failed to load model; please check the path: {e}")
    exit()

module_params = defaultdict(int)
total_params = [0]

if isinstance(data, (dict, OrderedDict)):
    analyze_nested_dict(data, module_params, total_params)
else:
    print(f"Error: this script only supports dict or OrderedDict files, but the loaded type is {type(data)}")
    exit()

print("=" * 60)
print("--- Model parameter analysis with nested-structure support ---")
if not module_params:
    print("Error: no model parameters (torch.Tensor) were found in the file. Please check the file contents.")
else:
    print("Parameter count by module:")
    for module_name, count in sorted(module_params.items()):
        print(f"  - module: {module_name:<30} | parameters: {count:>12,} (~{count/1e6:.2f}M)")

    print("-" * 60)
    print("Total:")
    final_total = total_params[0]
    print(f"  - Total model parameters:     {final_total:,} (~{final_total/1e6:.2f}M)")
print("=" * 60)