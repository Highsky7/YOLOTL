import torch
# Check GPU name (based on the graphics card connected to cuda:0)
print(torch.cuda.get_device_name(device = 0)) # e.g., 'NVIDIA TITAN X (Pascal)'

# Check the number of available GPUs
print(torch.cuda.device_count()) # e.g., 3