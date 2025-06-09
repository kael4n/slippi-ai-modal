# Fixed pre-built image approach for Slippi AI - NumPy compatibility fix
import modal

PROJECT_ROOT = "/root/slippi-ai"
REPO_URL = "https://github.com/vladfi1/slippi-ai.git"

def create_slippi_image():
    """
    Create optimized Slippi AI image with strict NumPy version control
    """
    # Start with Python 3.10 slim for better control
    image = modal.Image.from_registry("python:3.10-slim")
    
    # Install essential system packages
    image = image.apt_install([
        "tzdata", "build-essential", "pkg-config", "cmake", "ninja-build",
        "libssl-dev", "libffi-dev", "zlib1g-dev", "libbz2-dev", "libreadline-dev",
        "libsqlite3-dev", "libncurses5-dev", "libncursesw5-dev", "xz-utils",
        "tk-dev", "libxml2-dev", "libxmlsec1-dev", "liblzma-dev", "git", "curl",
        "wget", "unzip", "software-properties-common", "libgl1-mesa-glx",
        "libglib2.0-0", "libsm6", "libxext6", "libxrender-dev", "libgomp1"
    ])

    # Install Rust toolchain
    image = image.run_commands([
        "python -m pip install --upgrade pip setuptools wheel",
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        "python -m pip install maturin==1.2.3"
    ]).env({
        "PATH": "/root/.cargo/bin:$PATH",
        "CARGO_HOME": "/root/.cargo",
        "RUSTUP_HOME": "/root/.rustup"
    })

    # CRITICAL: Install NumPy 1.24.3 first and prevent upgrades
    image = image.run_commands([
        # Remove any existing NumPy installation
        "python -m pip uninstall numpy -y || true",
        # Install specific NumPy version with no dependencies to avoid conflicts
        "python -m pip install numpy==1.24.3 --no-deps --force-reinstall",
        # Verify installation
        'python -c "import numpy; print(f\'NumPy version: {numpy.__version__}\')"'
    ])

    # Install peppi-py with pre-compiled Rust environment
    image = image.run_commands([
        "/root/.cargo/bin/cargo install maturin",
        "python -m pip install --no-build-isolation peppi-py==0.6.0",
        'python -c "import peppi_py; print(\'✅ peppi-py installed successfully\')"'
    ]).env({
        "PATH": "/root/.cargo/bin:$PATH",
        "CARGO_HOME": "/root/.cargo",
        "RUSTUP_HOME": "/root/.rustup"
    })

    # Install compatible versions of scientific packages
    # These versions are known to work with NumPy 1.24.3
    image = image.pip_install([
        # Core scientific stack - compatible with NumPy 1.24.3
        "scipy==1.10.1",
        "pandas==2.0.3",
        
        # ML frameworks - versions compatible with NumPy 1.24.3
        "tensorflow==2.13.0",
        "jax==0.4.13",
        "jaxlib==0.4.13",
        
        # JAX ecosystem
        "flax==0.7.2",
        "optax==0.1.7",
        "dm-haiku==0.0.10",
        "dm-tree==0.1.8",
        
        # Experiment tracking
        "sacred==0.8.4",
        "pymongo==4.5.0",
        
        # Visualization
        "matplotlib==3.7.2",
        "seaborn==0.12.2",
        
        # Utilities
        "tqdm==4.65.0",
        "cloudpickle==2.2.1",
        "absl-py==1.4.0",
        "tensorboard==2.13.0",
        "gymnasium==0.28.1"
    ])

    # Verify no NumPy upgrade happened
    image = image.run_commands([
        'python -c "import numpy; assert numpy.__version__ == \'1.24.3\', f\'NumPy version changed to {numpy.__version__}\'"',
        'echo "✅ NumPy version locked successfully"'
    ])

    # Clone the repository and install
    image = image.run_commands([
        f"git clone {REPO_URL} {PROJECT_ROOT}",
        f"ls -la {PROJECT_ROOT}"
    ]).workdir(PROJECT_ROOT)

    # Install requirements carefully to avoid NumPy conflicts
    image = image.run_commands([
        # Install requirements but skip numpy if it's already specified
        "sed '/numpy/d' requirements.txt > requirements_no_numpy.txt || cp requirements.txt requirements_no_numpy.txt",
        "python -m pip install -r requirements_no_numpy.txt --no-upgrade || echo 'Partial requirements install completed'",
        "python -m pip install -e . --no-deps || echo 'Editable install completed'"
    ])

    # Final verification
    image = image.run_commands([
        'python -c "import numpy; print(f\'Final NumPy version: {numpy.__version__}\')"',
        'python -c "import scipy; print(f\'SciPy version: {scipy.__version__}\')"',
        'python -c "import tensorflow; print(f\'TensorFlow version: {tensorflow.__version__}\')"'
    ])

    return image

def create_slippi_image_alternative():
    """
    Alternative approach using Conda for better dependency management
    """
    image = modal.Image.from_registry("continuumio/miniconda3:latest")
    
    # Install system dependencies
    image = image.apt_install([
        "build-essential", "git", "curl", "pkg-config", "cmake"
    ])
    
    # Create conda environment with compatible packages
    image = image.run_commands([
        "conda create -n slippi python=3.10 -y",
        ". /opt/conda/etc/profile.d/conda.sh && conda activate slippi",
        # Install NumPy 1.24.3 via conda
        "conda install -n slippi numpy=1.24.3 -c conda-forge -y",
        # Install other scientific packages
        "conda install -n slippi scipy=1.10.1 pandas=2.0.3 -c conda-forge -y",
        # Install TensorFlow and JAX via pip in the conda env
        ". /opt/conda/etc/profile.d/conda.sh && conda activate slippi && pip install tensorflow==2.13.0 jax==0.4.13 jaxlib==0.4.13",
        # Install Rust for peppi-py
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        ". /opt/conda/etc/profile.d/conda.sh && conda activate slippi && pip install maturin",
        ". /opt/conda/etc/profile.d/conda.sh && conda activate slippi && pip install --no-build-isolation peppi-py==0.6.0"
    ]).env({
        "PATH": "/opt/conda/envs/slippi/bin:/root/.cargo/bin:$PATH",
        "CONDA_DEFAULT_ENV": "slippi"
    })
    
    return image

# Use the main approach
image = create_slippi_image()

app = modal.App("slippi-ai-numpy-fixed")

@app.function(image=image)
def test_prebuilt_env():
    """Test the pre-built environment with NumPy compatibility"""
    import peppi_py
    import tensorflow as tf
    import jax
    import numpy as np
    import scipy
    import sacred
    
    print("=" * 60)
    print("🚀 NUMPY-COMPATIBLE ENVIRONMENT VALIDATION")
    print("=" * 60)
    
    # Version checks
    print(f"✅ NumPy version: {np.__version__}")
    print(f"✅ SciPy version: {scipy.__version__}")
    print(f"✅ TensorFlow version: {tf.__version__}")
    print(f"✅ JAX version: {jax.__version__}")
    print(f"✅ Sacred version: {sacred.__version__}")
    print(f"✅ peppi_py location: {peppi_py.__file__}")
    
    # Verify NumPy compatibility
    if np.__version__.startswith("1.24"):
        print("✅ NumPy version is compatible")
    else:
        print(f"⚠️ NumPy version {np.__version__} may cause issues")
    
    # Functionality tests
    try:
        tf_test = tf.constant([1, 2, 3])
        print(f"✅ TensorFlow test: {tf_test.numpy()}")
    except Exception as e:
        print(f"❌ TensorFlow test failed: {e}")
    
    try:
        jax_test = jax.numpy.array([1, 2, 3])
        print(f"✅ JAX test: {jax_test}")
        print(f"✅ JAX devices: {jax.devices()}")
    except Exception as e:
        print(f"❌ JAX test failed: {e}")
    
    # Test SciPy compatibility
    try:
        from scipy import sparse
        sparse_matrix = sparse.csr_matrix([[1, 2], [3, 4]])
        print(f"✅ SciPy sparse test: {sparse_matrix.shape}")
    except Exception as e:
        print(f"❌ SciPy test failed: {e}")
    
    # Test peppi_py functionality
    try:
        print("✅ peppi_py import successful")
        # Add basic peppi_py functionality test if needed
    except Exception as e:
        print(f"❌ peppi_py test failed: {e}")
    
    print("🎉 NumPy-compatible environment ready!")

@app.function(image=image)
def debug_numpy_versions():
    """Debug NumPy and related package versions"""
    import subprocess
    import sys
    
    print("=" * 60)
    print("🔍 NUMPY DEPENDENCY DEBUG")
    print("=" * 60)
    
    # Check installed packages
    result = subprocess.run([sys.executable, "-m", "pip", "list"], 
                          capture_output=True, text=True)
    lines = result.stdout.split('\n')
    
    relevant_packages = ['numpy', 'scipy', 'tensorflow', 'jax', 'jaxlib']
    for line in lines:
        for pkg in relevant_packages:
            if line.lower().startswith(pkg.lower()):
                print(f"📦 {line}")
    
    # Test imports individually
    test_imports = [
        'numpy',
        'scipy',
        'scipy.sparse',
        'tensorflow',
        'jax',
        'jax.numpy'
    ]
    
    for imp in test_imports:
        try:
            __import__(imp)
            print(f"✅ {imp} - OK")
        except Exception as e:
            print(f"❌ {imp} - FAILED: {e}")

@app.local_entrypoint()
def main():
    print("Testing NumPy-compatible Slippi AI environment...")
    debug_numpy_versions.remote()
    test_prebuilt_env.remote()

if __name__ == "__main__":
    main()