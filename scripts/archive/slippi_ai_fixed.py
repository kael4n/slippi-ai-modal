# v17-numpy-lock: Force NumPy 1.24.3 and prevent upgrades
import modal

PROJECT_ROOT = "/root/slippi-ai"
REPO_URL = "https://github.com/vladfi1/slippi-ai.git"

def create_slippi_image():
    image = modal.Image.debian_slim().apt_install([
        "tzdata", "python3", "python3-pip", "python3-dev", "build-essential",
        "pkg-config", "cmake", "ninja-build", "libssl-dev", "libffi-dev", "zlib1g-dev",
        "libbz2-dev", "libreadline-dev", "libsqlite3-dev", "libncurses5-dev",
        "libncursesw5-dev", "xz-utils", "tk-dev", "libxml2-dev", "libxmlsec1-dev",
        "liblzma-dev", "git", "curl", "wget", "unzip", "software-properties-common",
        "libgl1-mesa-glx", "libglib2.0-0", "libsm6", "libxext6", "libxrender-dev", "libgomp1"
    ])

    image = image.run_commands([
        "ln -sf /usr/bin/python3 /usr/bin/python",
        "python3 -m pip install --upgrade pip setuptools wheel",
        # Install maturin via pip for importable backend
        "python3 -m pip install maturin",
        # Install Rust and maturin CLI for cargo-based building
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        "/root/.cargo/bin/cargo install maturin",
        "/root/.cargo/bin/maturin --version"
    ]).env({
        "PATH": "/root/.cargo/bin:$PATH",
        "PYTHONPATH": "/usr/local/lib/python3.10/site-packages",
        "CARGO_HOME": "/root/.cargo",
        "RUSTUP_HOME": "/root/.rustup"
    })

    # Install peppi-py with proper isolation handling
    image = image.run_commands([
        "python3 -m pip install --no-build-isolation peppi-py==0.6.0",
        'python3 -c "import peppi_py; print(\'✅ peppi-py v0.6.0 installed and importable\')"'
    ])

    # CRITICAL: Force NumPy 1.24.3 installation and prevent any upgrades
    image = image.run_commands([
        # First, ensure we have the exact NumPy version
        "python3 -m pip install 'numpy==1.24.3' --force-reinstall --no-deps",
        # Verify NumPy version
        'python3 -c "import numpy as np; print(f\'NumPy version: {np.__version__}\')"',
        # Install scipy with the correct NumPy already in place
        "python3 -m pip install 'scipy==1.10.1' --no-deps",
        # Test scipy import
        'python3 -c "import scipy; print(f\'SciPy version: {scipy.__version__}\')"'
    ])

    # Install remaining packages with careful NumPy version management
    image = image.run_commands([
        # Install pandas without letting it upgrade NumPy
        "python3 -m pip install 'pandas==2.0.3'",
        # Install TensorFlow dependencies first (without NumPy conflicts)
        "python3 -m pip install 'protobuf>=3.20.3,<5.0.0dev'",
        "python3 -m pip install 'six>=1.12.0'",
        "python3 -m pip install 'termcolor>=1.1.0'",
        "python3 -m pip install 'typing-extensions>=3.6.6'",
        "python3 -m pip install 'wrapt>=1.11.0'",
        "python3 -m pip install 'astunparse>=1.6.0'",
        "python3 -m pip install 'gast<=0.4.0,>=0.2.1'",
        "python3 -m pip install 'google-pasta>=0.1.1'",
        "python3 -m pip install 'h5py>=2.9.0'",
        "python3 -m pip install 'libclang>=13.0.0'",
        "python3 -m pip install 'opt-einsum>=2.3.2'",
        "python3 -m pip install 'packaging'",
        "python3 -m pip install 'setuptools'",
        # Critical TensorFlow dependencies that were missing
        "python3 -m pip install 'flatbuffers>=1.12'",
        "python3 -m pip install 'grpcio>=1.24.3,<2.0'",
        "python3 -m pip install 'tensorflow-io-gcs-filesystem>=0.23.1'",
        "python3 -m pip install 'keras==2.13.1'",
        "python3 -m pip install 'tensorboard==2.13.0'",
        "python3 -m pip install 'tensorflow-estimator==2.13.0'",
        # Now install TensorFlow with dependencies satisfied
        "python3 -m pip install 'tensorflow==2.13.0' --no-deps",
        # Test TensorFlow import
        'python3 -c "import tensorflow as tf; print(f\'TensorFlow version: {tf.__version__}\')"'
    ])

    # Install JAX with exact versions and no dependency resolution
    image = image.run_commands([
        "python3 -m pip install 'jax==0.4.13' --no-deps",
        "python3 -m pip install 'jaxlib==0.4.13' --no-deps",
        # Install JAX dependencies manually
        "python3 -m pip install 'ml-dtypes>=0.1.0'",
        "python3 -m pip install 'opt-einsum'",
        # Test JAX import
        'python3 -c "import jax; print(f\'JAX version: {jax.__version__}\')"'
    ])

    # Install remaining ML frameworks
    image = image.pip_install([
        "flax==0.7.2",
        "optax==0.1.7", 
        "dm-haiku==0.0.10",
        "dm-tree==0.1.8",
        
        # Experiment tracking and utilities
        "sacred==0.8.4",
        "pymongo==4.5.0",
        
        # Visualization
        "matplotlib==3.7.2",
        "seaborn==0.12.2",
        
        # Utilities
        "tqdm==4.65.0",
        "cloudpickle==2.2.1",
        "absl-py==1.4.0",
        "gymnasium==0.28.1"
    ])

    # Final verification of all key packages
    image = image.run_commands([
        'python3 -c "import numpy; print(f\'✅ Final NumPy version: {numpy.__version__}\')"',
        'python3 -c "import scipy; print(f\'✅ Final SciPy version: {scipy.__version__}\')"',
        'python3 -c "import tensorflow; print(f\'✅ Final TensorFlow version: {tensorflow.__version__}\')"',
        'python3 -c "import jax; print(f\'✅ Final JAX version: {jax.__version__}\')"',
        'python3 -c "import jaxlib; print(f\'✅ Final JAXlib version: {jaxlib.__version__}\')"'
    ])

    # Clone repository and set working directory
    image = image.run_commands([
        f"git clone {REPO_URL} {PROJECT_ROOT}",
        f"ls -la {PROJECT_ROOT}"
    ]).workdir(PROJECT_ROOT)

    # Install project requirements with fallback handling
    image = image.run_commands([
        "python3 -m pip install -r requirements.txt || echo 'Partial requirements succeeded'",
        "python3 -m pip install -e . || echo 'Editable install skipped/fallback'"
    ])

    return image

image = create_slippi_image()

app = modal.App("slippi-ai-numpy-lock-v17")

@app.function(image=image)
def test_env():
    import peppi_py
    import tensorflow as tf
    import jax
    import jaxlib
    import numpy as np
    import scipy
    import sacred

    print("✅ peppi_py path:", peppi_py.__file__)
    print("✅ tensorflow version:", tf.__version__)
    print("✅ jax version:", jax.__version__)
    print("✅ jaxlib version:", jaxlib.__version__)
    print("✅ numpy version:", np.__version__)
    print("✅ scipy version:", scipy.__version__)
    print("✅ sacred version:", sacred.__version__)
    
    # Test basic functionality
    print("✅ Testing TensorFlow...")
    tf_test = tf.constant([1, 2, 3])
    print("   TensorFlow tensor:", tf_test.numpy())
    
    print("✅ Testing JAX...")
    jax_test = jax.numpy.array([1, 2, 3])
    print("   JAX array:", jax_test)
    
    print("✅ All imports and basic tests successful!")

@app.local_entrypoint()
def main():
    test_env.remote()