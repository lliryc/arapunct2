import torch

# Check if CUDA is available before checking device capabilities
if torch.cuda.is_available():
    # Check if GPU benefits from bfloat16
    try:
        if torch.cuda.get_device_capability()[0] >= 8:
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16
    except Exception as e:
        print(f"Warning: Error checking CUDA capabilities: {e}")
        print("Defaulting to float16")
        torch_dtype = torch.float16
else:
    print("CUDA not available, using CPU with float32")
    torch_dtype = torch.float32