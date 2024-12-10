import torch
print(torch.cuda.is_available())
print(torch.ops._C.rms_norm)  # Ensure it does not raise NotImplementedError


