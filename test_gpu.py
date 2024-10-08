import torch

if torch.cuda.is_available():
    print("available GPU")
    num_gpus = torch.cuda.device_count()
    print(f"Số lượng GPU: {num_gpus}")
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"GPU {i}: {gpu_name}")
else:
    print("not available GPU")