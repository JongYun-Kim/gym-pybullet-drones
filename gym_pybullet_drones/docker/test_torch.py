import sys
import torch
import os
from termcolor import cprint

# Check the environment for PyTorch and GPU availability
def check_environment():
    # Print Python Version
    python_version = sys.version
    cprint(f"Python Version: {python_version}", "green")

    # Print PyTorch Version
    pytorch_version = torch.__version__
    cprint(f"PyTorch Version: {pytorch_version}", "green")

    # Print Current Directory
    current_directory = os.getcwd()
    cprint(f"Current Directory: {current_directory}", "cyan")

    # Print Python Executable Directory
    python_executable = sys.executable
    cprint(f"Python Executable Directory: {python_executable}", "cyan")

    # Check GPU Availability
    gpu_available = torch.cuda.is_available()
    cprint(f"Is GPU Available: {gpu_available}", "yellow")

    # Print CUDA Version (if available)
    if gpu_available:
        cuda_version = torch.version.cuda
        cprint(f"CUDA Version: {cuda_version}", "blue")

        # Print the number of GPUs
        gpu_count = torch.cuda.device_count()
        cprint(f"Number of GPUs: {gpu_count}", "yellow")

        # Print GPU Names
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            cprint(f"GPU {i}: {gpu_name}", "magenta")
    else:
        cprint("No GPUs available.", "red")


    # Check cuDNN version
    cudnn_version = torch.backends.cudnn.version()
    cprint(f"cuDNN Version: {cudnn_version}", "blue")

    # Check current GPU device (if GPU is available)
    if gpu_available:
        current_gpu = torch.cuda.current_device()
        cprint(f"Current GPU Device ID: {current_gpu}", "yellow")
    else:
        cprint("GPU not available: Cannot determine current GPU device.", "red")

    # Check NCCL library version (responsible for inter-GPU communication)
    if gpu_available:
        nccl_version = torch.cuda.nccl.version()
        cprint(f"NCCL Version: {nccl_version}", "magenta")
    else:
        cprint("GPU not available: Cannot determine NCCL version.", "red")

    # Re-check PyTorch version (should be 2.6.0+cu124 for GPU support, not a CPU-only build)
    cprint(f"Re-checked PyTorch Version: {torch.__version__}", "green")


if __name__ == "__main__":
    check_environment()