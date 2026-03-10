import os
import glob
import torch
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension
from setuptools import find_packages, setup

requirements = ["torch", "torchvision"]

def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "src")

    # Collect source files
    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    sources = main_file + source_cpu
    extension = CppExtension
    extra_compile_args = {"cxx": []}
    define_macros = []

    # Explicitly set CUDA paths
    cuda_root = "/usr/local/apps/cuda/12.1"
    cuda_include_dir = os.path.join(cuda_root, "targets/x86_64-linux/include")
    cuda_lib_dir = os.path.join(cuda_root, "targets/x86_64-linux/lib")
    
    print(f"Using CUDA root: {cuda_root}")
    print(f"Using CUDA include: {cuda_include_dir}")
    print(f"Using CUDA lib: {cuda_lib_dir}")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")

    # Force CUDA extension without availability check
    extension = CUDAExtension
    sources += source_cuda
    define_macros += [("WITH_CUDA", None)]
    
    # Add debugging flags and paths
    extra_compile_args["nvcc"] = [
        "-DCUDA_HAS_FP16=1",
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
        "-Xcompiler=-fPIC",  # Ensure position-independent code
        f"-I{cuda_include_dir}",  # Explicitly add CUDA include path
    ]
    
    extra_compile_args["cxx"] = ["-fPIC"]  # Ensure position-independent code for C++

    sources = [os.path.join(extensions_dir, s) for s in sources]
    include_dirs = [
        extensions_dir,
        cuda_include_dir,
        os.path.join(cuda_root, "include"),  # Add both possible include paths
    ]
    
    library_dirs = [
        cuda_lib_dir,
        os.path.join(cuda_root, "lib64"),  # Add both possible library paths
    ]

    ext_modules = [
        extension(
            name="MultiScaleDeformableAttention",
            sources=sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            library_dirs=library_dirs,
            libraries=["cudart"]  # Explicitly link against CUDA runtime
        )
    ]
    return ext_modules

if __name__ == '__main__':
    # Print PyTorch configuration info
    print("\nPyTorch Configuration:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print("\nStarting build...\n")
    
    setup(
        name="MultiScaleDeformableAttention",
        version="1.0",
        author="Weijie Su",
        url="https://github.com/fundamentalvision/Deformable-DETR",
        description="PyTorch Wrapper for CUDA Functions of Multi-Scale Deformable Attention",
        packages=find_packages(exclude=("configs", "tests",)),
        ext_modules=get_extensions(),
        cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},  # Disable ninja builder
    )