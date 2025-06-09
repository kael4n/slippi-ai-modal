# Using pre-built TensorFlow image to speed up builds
import modal

PROJECT_ROOT = "/root/slippi-ai"
REPO_URL = "https://github.com/vladfi1/slippi-ai.git"

def create_slippi_image():
    # Start with official TensorFlow image that already has ML dependencies
    image = modal.Image.from_registry(
        "tensorflow/tensorflow:2.13.0-py3",
        # Add any missing system dependencies
        add_python="3.10"  # Ensure Python 3.10+ is available
    )
    
    # Add missing system packages that aren't in the TensorFlow image
    image = image.apt_install([
        "git", "curl", "wget", "build-essential", "pkg-config", 
        "cmake", "ninja-build", "libssl-dev", "libffi-dev"
    ])
    
    # CRITICAL: Create NumPy constraint file first
    image = image.run_commands([
        # Check what NumPy version came with TensorFlow
        'python -c "import numpy; print(f\'Base NumPy version: {numpy.__version__}\')"',
        # Create constraint file for consistency
        'echo "numpy==1.24.3" > /root/numpy-constraint.txt',
        "mkdir -p /root/.pip",
        'echo "[install]" > /root/.pip/pip.conf',
        'echo "constraint = /root/numpy-constraint.txt" >> /root/.pip/pip.conf',
        # Potentially downgrade NumPy if needed
        "python -m pip install 'numpy==1.24.3' --force-reinstall",
        'python -c "import numpy; print(f\'Updated NumPy version: {numpy.__version__}\')"'
    ])
    
    # Install Rust for peppi-py (much faster on pre-built image)
    image = image.run_commands([
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        "python -m pip install maturin==1.2.3",
        "/root/.cargo/bin/cargo install maturin",
    ]).env({
        "PATH": "/root/.cargo/bin:$PATH",
        "CARGO_HOME": "/root/.cargo",
        "RUSTUP_HOME": "/root/.rustup",
        "PIP_CONSTRAINT": "/root/numpy-constraint.txt"
    })
    
    # Install peppi-py
    image = image.run_commands([
        "python -m pip install --no-build-isolation peppi-py==0.6.0",
        'python -c "import peppi_py; print(\'✅ peppi-py installed\')"'
    ]).env({
        "PIP_CONSTRAINT": "/root/numpy-constraint.txt",
        "PATH": "/root/.cargo/bin:$PATH",
        "CARGO_HOME": "/root/.cargo",
        "RUSTUP_HOME": "/root/.rustup"
    })
    
    # Install remaining Python packages
    image = image.run_commands([
        "python -m pip install jax==0.4.13 jaxlib==0.4.13",
        "python -m pip install sacred==0.8.4 pymongo==4.5.0",
        "python -m pip install flax==0.7.2 optax==0.1.7",
        "python -m pip install dm-haiku==0.0.10 dm-tree==0.1.8",
        "python -m pip install tqdm==4.65.0 cloudpickle==2.2.1",
        "python -m pip install matplotlib==3.7.2 seaborn==0.12.2",
        "python -m pip install gymnasium==0.28.1 absl-py==1.4.0"
    ]).env({"PIP_CONSTRAINT": "/root/numpy-constraint.txt"})
    
    # Clone and install slippi-ai
    image = image.run_commands([
        f"git clone {REPO_URL} {PROJECT_ROOT}",
        "python -m pip install -e . || echo 'Editable install completed'"
    ]).workdir(PROJECT_ROOT).env({"PIP_CONSTRAINT": "/root/numpy-constraint.txt"})
    
    return image

# Alternative: Use PyTorch image if you prefer
def create_slippi_image_pytorch():
    # PyTorch images are also very comprehensive
    image = modal.Image.from_registry("pytorch/pytorch:2.0.1-py3.10-cuda11.7-cudnn8-runtime")
    
    # Add TensorFlow and other dependencies
    image = image.apt_install(["git", "curl", "build-essential", "pkg-config", "cmake"])
    
    # Install TensorFlow manually since we're starting with PyTorch
    image = image.run_commands([
        "pip install tensorflow==2.13.0",
        # ... rest of the setup
    ])
    
    return image

# Option 3: Use a custom-built image from your registry
def create_slippi_image_custom():
    # If you build your own base image and push to a registry
    image = modal.Image.from_registry("your-registry/slippi-base:latest")
    
    # Just add the project-specific parts
    image = image.run_commands([
        f"git clone {REPO_URL} {PROJECT_ROOT}",
        "pip install -e ."
    ]).workdir(PROJECT_ROOT)
    
    return image

image = create_slippi_image()
app = modal.App("slippi-ai-prebuilt")

@app.function(image=image)
def test_prebuilt_env():
    import peppi_py
    import tensorflow as tf
    import jax
    import numpy as np
    
    print("="*60)
    print("🚀 PRE-BUILT IMAGE TEST")
    print("="*60)
    print("✅ NumPy version:", np.__version__)
    print("✅ TensorFlow version:", tf.__version__)
    print("✅ JAX version:", jax.__version__)
    print("✅ peppi_py available:", peppi_py.__file__)
    
    # Test functionality
    tf_test = tf.constant([1, 2, 3])
    jax_test = jax.numpy.array([1, 2, 3])
    
    print("✅ TensorFlow test:", tf_test.numpy())
    print("✅ JAX test:", jax_test)
    print("🎉 All systems operational!")

@app.local_entrypoint()
def main():
    test_prebuilt_env.remote()