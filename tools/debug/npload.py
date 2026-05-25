import numpy as np

mean_data = np.load('./dataset/humanml_spatial_norm/Mean_raw.npy')
std_data = np.load('dataset/humanml_spatial_norm/Std_raw.npy')
print(mean_data)
print("Mean data shape:", mean_data.shape)
print("Mean data type:", mean_data.dtype)

print("*"*100)

print(std_data)
print("Std data shape:", std_data.shape)
print("Std data type:", std_data.dtype)

