import numpy as np

npy_path = './dataset/HumanML3D/new_joints/010642.npy'

data = np.load(npy_path, allow_pickle=True)

if isinstance(data, np.ndarray) and data.dtype == object:
    data = data.item()
    print("Keys in the .npy file:", data.keys())
    
    for key, value in data.items():
        print(f"\nKey: {key}")
        print(f"Type: {type(value)}")
        if isinstance(value, (np.ndarray, list)):
            print(f"Shape: {np.array(value).shape}")
        print(f"Content: {value}")
else:
    print("The .npy file contains an array, not a dictionary.")
    print("Shape of the array:", data.shape)
    print("Array content:", data)