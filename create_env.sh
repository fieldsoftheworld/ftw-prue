#!/bin/bash

# Exit on error
set -e

# Default configuration
ENV_NAME="prue-m2f"
ENV_PATH_OVERRIDE=""
CUDA_HOME_OVERRIDE=""
CUDA_ARCH_LIST_OVERRIDE=""

usage() {
    cat <<'EOF'
Usage: create_env.sh [--env-name <name>] [--env-path </custom/path>] \
                     [--cuda-home </usr/local/cuda>] [--cuda-arch-list "8.0"]

Options:
  --env-name         Name to use under the default conda prefix (default: prue-m2f)
  --env-path         Fully-qualified path for the environment (overrides --env-name)
  --cuda-home        Path to CUDA toolkit (defaults to existing CUDA_HOME or nvcc location)
  --cuda-arch-list   Value for TORCH_CUDA_ARCH_LIST (e.g., "7.0" for V100, "9.0" for H100)

Examples:
  ./create_env.sh --cuda-arch-list "8.0"
  ./create_env.sh --env-path $HOME/miniconda3/envs/prue-m2f --cuda-home /usr/local/cuda-12.1
EOF
}

# Argument parsing
while [ $# -gt 0 ]; do
    case "$1" in
        --env-name)
            ENV_NAME="$2"
            shift 2
            ;;
        --env-path)
            ENV_PATH_OVERRIDE="$2"
            shift 2
            ;;
        --cuda-home)
            CUDA_HOME_OVERRIDE="$2"
            shift 2
            ;;
        --cuda-arch-list)
            CUDA_ARCH_LIST_OVERRIDE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"

echo "=========================================="
echo "Setting up environment in ${PROJECT_DIR}"
echo "=========================================="

# Determine Conda base
if ! command -v conda >/dev/null 2>&1; then
    echo "Error: conda command not found. Please install Miniconda/Anaconda first."
    exit 1
fi
CONDA_BASE="$(conda info --base)"

if [ -n "$ENV_PATH_OVERRIDE" ]; then
    ENV_PATH="$ENV_PATH_OVERRIDE"
else
    ENV_PATH="${CONDA_BASE}/envs/${ENV_NAME}"
fi

# Resolve CUDA toolkit location
if [ -n "$CUDA_HOME_OVERRIDE" ]; then
    CUDA_HOME="$CUDA_HOME_OVERRIDE"
elif [ -n "$CUDA_HOME" ]; then
    CUDA_HOME="$CUDA_HOME"
elif command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
else
    CUDA_HOME=""
fi

if [ -n "$CUDA_HOME" ]; then
    export CUDA_HOME
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
else
    echo "Warning: CUDA_HOME is not set and nvcc was not found in PATH."
    echo "         The Mask2Former install docs require CUDA_HOME for building MSDeformAttn."
fi

# Determine CUDA arch list
if [ -n "$CUDA_ARCH_LIST_OVERRIDE" ]; then
    export TORCH_CUDA_ARCH_LIST="$CUDA_ARCH_LIST_OVERRIDE"
elif [ -z "$TORCH_CUDA_ARCH_LIST" ]; then
    export TORCH_CUDA_ARCH_LIST="8.0"
fi

# Verify CUDA setup
echo "Checking CUDA setup..."
if command -v nvcc >/dev/null 2>&1; then
    which nvcc
    nvcc --version
else
    echo "nvcc not found on PATH; ensure CUDA toolkit is installed if you plan to build GPU ops."
fi
echo "CUDA_HOME: ${CUDA_HOME:-unset}"
echo "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}"

# Initialize conda for bash
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda deactivate 2>/dev/null || true
hash -r # Clear hash table

echo "Creating conda environment at: ${ENV_PATH}"

# Check if environment already exists
if [ -d "${ENV_PATH}" ]; then
    echo "Warning: Environment already exists at ${ENV_PATH}"
    echo "Removing existing environment to create a fresh one..."
    rm -rf "${ENV_PATH}"
fi

# Create and activate new conda virtual environment
conda create -p "${ENV_PATH}" python=3.11 -y
eval "$(conda shell.bash hook)"
conda activate "${ENV_PATH}"

# Install numpy 1.26.4 first to prevent PyTorch from pulling in numpy 2.x
echo "Installing numpy 1.26.4 (required for PyTorch 2.1.0 compatibility)..."
pip install "numpy==1.26.4"

# Install PyTorch with CUDA 12.1 support
echo "Installing PyTorch..."
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Ensure numpy stays at 1.26.4 (PyTorch 2.1.0 requires numpy < 2.0)
echo "Ensuring numpy 1.26.4 is installed (PyTorch 2.1.0 compatibility)..."
pip install --force-reinstall "numpy==1.26.4"

# Verify PyTorch installation before proceeding
echo "Verifying PyTorch installation..."
python -c "import torch; import numpy; print('PyTorch version:', torch.__version__); print('NumPy version:', numpy.__version__); print('CUDA available:', torch.cuda.is_available())" || {
    echo "Error: PyTorch installation failed or cannot be imported!"
    exit 1
}

# Set FORCE_CUDA=1 to allow CUDA extension builds even if CUDA runtime is not detected during build
# This is needed because during the build process, CUDA runtime might not be accessible
# even though CUDA_HOME is properly set
export FORCE_CUDA=1
echo "Set FORCE_CUDA=1 for CUDA extension builds"

# Install ninja and cmake for faster builds
echo "Installing build tools..."
conda install -y ninja cmake

# Install the project and all optional dependency groups from pyproject.toml
echo "Installing ftw-prue and optional dependencies from pyproject.toml..."
cd "${PROJECT_DIR}"
pip install -e ".[all]"

# Install opencv
echo "Installing OpenCV..."
conda install -y -c conda-forge opencv

# Install Detectron2 from local directory
# Use --no-build-isolation so detectron2's setup.py can access the installed torch
echo "Installing Detectron2 from local directory..."
DETECTRON2_DIR="${PROJECT_DIR}/vendor/detectron2"
if [ ! -d "${DETECTRON2_DIR}" ]; then
    echo "Error: detectron2 directory not found at ${DETECTRON2_DIR}"
    exit 1
fi
cd "${DETECTRON2_DIR}"
pip install --no-build-isolation -e .

# Verify detectron2 installation
echo "Verifying Detectron2 installation..."
python -c "import detectron2; print('Detectron2 version:', detectron2.__version__)" || {
    echo "Error: Detectron2 installation failed!"
    exit 1
}
python -c "import detectron2.utils.comm; print('Detectron2 utils.comm imported successfully')" || {
    echo "Error: Cannot import detectron2.utils.comm!"
    exit 1
}

# Install panopticapi from local directory
echo "Installing panopticapi from local directory..."
PANOPTICAPI_DIR="${PROJECT_DIR}/vendor/panopticapi"
if [ ! -d "${PANOPTICAPI_DIR}" ]; then
    echo "Error: panopticapi directory not found at ${PANOPTICAPI_DIR}"
    exit 1
fi
cd "${PANOPTICAPI_DIR}"
pip install -e .

# Build CLUSTEN custom CUDA kernel
echo "Building CLUSTEN custom CUDA kernel..."
CLUSTEN_DIR="${PROJECT_DIR}/vendor/mask2former/modeling/clusten/src"
if [ -d "${CLUSTEN_DIR}" ]; then
    cd "${CLUSTEN_DIR}"
    python setup.py build_ext --inplace
    
    # Create a setup_path.py file in the clusten directory to help with imports
    cd "${PROJECT_DIR}/vendor/mask2former/modeling/clusten"
    cat > setup_path.py << 'EOF'
import os
import sys

# Add the src directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
if src_dir not in sys.path:
    sys.path.append(src_dir)
EOF
else
    echo "Warning: CLUSTEN directory not found, skipping CLUSTEN build"
fi

# Build Mask2Former pixel decoder operations (MSDeformAttn)
echo "Building Mask2Former pixel decoder operations (TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST})..."
PIXEL_DECODER_DIR="${PROJECT_DIR}/vendor/mask2former/modeling/pixel_decoder/ops"
if [ -d "${PIXEL_DECODER_DIR}" ]; then
    cd "${PIXEL_DECODER_DIR}"
    python setup.py build install
else
    echo "Warning: Pixel decoder ops directory not found, skipping build"
fi

# Install geospatial packages
echo "Installing geospatial packages..."
conda install -y -c conda-forge rasterio gdal

# Verify CUDA and PyTorch versions
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"

python -c "import detectron2; print('Detectron2 installed successfully')"
python -c "import panopticapi; print('PanopticAPI installed successfully')"

echo "=========================================="
echo "Environment setup complete!"
echo "Environment path: ${ENV_PATH}"
echo "CUDA_HOME: ${CUDA_HOME:-unset}"
echo "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}"
echo "=========================================="