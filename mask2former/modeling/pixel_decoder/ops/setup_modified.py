# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/fundamentalvision/Deformable-DETR

import os
import glob

import torch

# from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CUDA_HOME as TORCH_CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension

from setuptools import find_packages
from setuptools import setup

requirements = ["torch", "torchvision"]

def get_cuda_home():
    # First check the environment variable
    cuda_home = os.environ.get('CUDA_HOME')
    
    if cuda_home is None:
        # Then check TORCH_CUDA_HOME
        cuda_home = TORCH_CUDA_HOME
        
    if cuda_home is None:
        # Finally, try to find it in some common locations
        common_paths = [
            '/usr/local/cuda',
            '/usr/local/apps/cuda/12.1',
            '/opt/cuda',
        ]
        for path in common_paths:
            if os.path.exists(path):
                cuda_home = path
                break
    
    # If CUDA_HOME points to a directory with symlinks, resolve to the actual paths
    if cuda_home and os.path.exists(cuda_home):
        include_dir = os.path.join(cuda_home, 'include')
        lib64_dir = os.path.join(cuda_home, 'lib64')
        
        # If these are symlinks, resolve them
        if os.path.islink(include_dir):
            real_include = os.path.join(cuda_home, os.readlink(include_dir))
            if os.path.exists(real_include):
                include_dir = real_include
                
        if os.path.islink(lib64_dir):
            real_lib64 = os.path.join(cuda_home, os.readlink(lib64_dir))
            if os.path.exists(real_lib64):
                lib64_dir = real_lib64
                
        return {
            'home': cuda_home,
            'include': include_dir,
            'lib64': lib64_dir
        }
    
    return None

def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "src")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    sources = main_file + source_cpu
    extension = CppExtension
    extra_compile_args = {"cxx": []}
    define_macros = []

    cuda_info = get_cuda_home()

    if cuda_info is None:
        raise NotImplementedError(
            'Could not find CUDA. Tried environment variable CUDA_HOME, '
            'PyTorch CUDA_HOME, and common install locations. '
            'Please either install CUDA in a standard location, '
            'or set CUDA_HOME environment variable.'
        )

    # # Force cuda since torch ask for a device, not if cuda is in fact available.
    # if (os.environ.get('FORCE_CUDA') or torch.cuda.is_available()) and CUDA_HOME is not None:
    #     extension = CUDAExtension
    #     sources += source_cuda
    #     define_macros += [("WITH_CUDA", None)]
    #     extra_compile_args["nvcc"] = [
    #         "-DCUDA_HAS_FP16=1",
    #         "-D__CUDA_NO_HALF_OPERATORS__",
    #         "-D__CUDA_NO_HALF_CONVERSIONS__",
    #         "-D__CUDA_NO_HALF2_OPERATORS__",
    #     ]
    # else:
    #     if CUDA_HOME is None:
    #         raise NotImplementedError('CUDA_HOME is None. Please set environment variable CUDA_HOME.')
    #     else:
    #         raise NotImplementedError('No CUDA runtime is found. Please set FORCE_CUDA=1 or test it by running torch.cuda.is_available().')

    # Set CUDA_HOME to the resolved path for PyTorch's benefit
    os.environ['CUDA_HOME'] = cuda_info['home']

    # Force cuda since torch ask for a device, not if cuda is in fact available.
    if (os.environ.get('FORCE_CUDA', '1') == '1' or torch.cuda.is_available()):
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

        # Add CUDA include paths
        include_dirs = [
            extensions_dir,
            cuda_info['include']
        ]
        
        # Add CUDA library paths
        library_dirs = [
            cuda_info['lib64']
        ]
        print(f"Using CUDA include path: {cuda_info['include']}")
        print(f"Using CUDA library path: {cuda_info['lib64']}")
    else:
        raise NotImplementedError(
            'CUDA is required. Please make sure CUDA is installed and '
            'torch.cuda.is_available() returns True.'
        )

    sources = [os.path.join(extensions_dir, s) for s in sources]

    # include_dirs = [extensions_dir]
    # ext_modules = [
    #     extension(
    #         "MultiScaleDeformableAttention",
    #         sources,
    #         include_dirs=include_dirs,
    #         define_macros=define_macros,
    #         extra_compile_args=extra_compile_args,
    #     )
    # ]

    ext_modules = [
        extension(
            "MultiScaleDeformableAttention",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            library_dirs=library_dirs,
        )
    ]
    return ext_modules

setup(
    name="MultiScaleDeformableAttention",
    version="1.0",
    author="Weijie Su",
    url="https://github.com/fundamentalvision/Deformable-DETR",
    description="PyTorch Wrapper for CUDA Functions of Multi-Scale Deformable Attention",
    packages=find_packages(exclude=("configs", "tests",)),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
