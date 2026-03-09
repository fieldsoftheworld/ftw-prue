import psutil
import torch
import resource

def log_system_info():
    process = psutil.Process()
    print(f"System RAM: {psutil.virtual_memory().available / (1024 * 1024 * 1024):.2f}GB available")
    print(f"Process Memory: {process.memory_info().rss / (1024 * 1024 * 1024):.2f}GB")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i} Memory: {torch.cuda.memory_allocated(i) / (1024 * 1024 * 1024):.2f}GB allocated")

def print_resource_limits():
    print(f"Max memory limit: {resource.getrlimit(resource.RLIMIT_AS)[0] / (1024 * 1024 * 1024):.2f}GB")
    print(f"Max processes: {resource.getrlimit(resource.RLIMIT_NPROC)}")