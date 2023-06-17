# importing torch  
import torch  
# get index of currently selected device  
torch.cuda.current_device() # returns 0 in my case  
# get number of GPUs available  
torch.cuda.device_count() # returns 1 in my case  
# get the name of the device  
torch.cuda.get_device_name(0) # 'NVIDIA GeForce RTX 3060 Laptop GPU' in my case
