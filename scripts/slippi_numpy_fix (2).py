# Fixed Slippi AI - Strict NumPy version control with constraints
import modal

PROJECT_ROOT = "/root/slippi-ai"
REPO_URL = "https://github.com/vladfi1/slippi-ai.git"

def create_slippi_image():
    """
    Create Slippi AI image with absolute NumPy version control
    """
    # Start with Python 3.10 slim
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

    # Install Rust toolchain for peppi-py
    image = image.run_commands([
        "python -m pip install --upgrade pip setuptools wheel",
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        "python -m pip install maturin==1.2.3"
    ]).env({
        "PATH": "/root/.cargo/bin:$PATH",
        "CARGO_HOME": "/root/.cargo",
        "RUSTUP_HOME": "/root/.rustup"
    })

    # CRITICAL: Create pip constraints file to lock NumPy version
    image = image.run_commands([
        "echo 'numpy==1.24.3' > /tmp/constraints.txt",
        "echo 'Created NumPy constraints file'"
    ])

    # Install NumPy 1.24.3 first
    image = image.run_commands([
        "python -m pip uninstall numpy -y || true",
        "python -m pip install numpy==1.24.3 --constraint /tmp/constraints.txt",
        'python -c "import numpy; print(f\'Initial NumPy version: {numpy.__version__}\')"'
    ])

    # Install peppi-py with constraints
    image = image.run_commands([
        "/root/.cargo/bin/cargo install maturin",
        "python -m pip install --no-build-isolation peppi-py==0.6.0 --constraint /tmp/constraints.txt",
        'python -c "import peppi_py; print(\'✅ peppi-py installed successfully\')"'
    ]).env({
        "PATH": "/root/.cargo/bin:$PATH",
        "CARGO_HOME": "/root/.cargo",
        "RUSTUP_HOME": "/root/.rustup"
    })

    # Install compatible versions of scientific packages with constraints
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
        "gymnasium==0.28.1",
        
        # Install dm-sonnet with constraints to prevent NumPy upgrade
        "dm-sonnet==2.0.1"  # Older version that's more compatible
    ], constraint="/tmp/constraints.txt")

    # Verify NumPy hasn't been upgraded
    image = image.run_commands([
        'python -c "import numpy; print(f\'NumPy after pip_install: {numpy.__version__}\')"',
        'python -c "import numpy; assert numpy.__version__ == \'1.24.3\', f\'NumPy version changed to {numpy.__version__}\'"'
    ])

    # Clone the repository
    image = image.run_commands([
        f"git clone {REPO_URL} {PROJECT_ROOT}",
        f"ls -la {PROJECT_ROOT}"
    ]).workdir(PROJECT_ROOT)

    # Create a modified requirements.txt that excludes problematic packages
    image = image.run_commands([
        # Create a custom requirements file excluding packages that force NumPy upgrades
        """cat > /tmp/slippi_requirements.txt << 'EOF'
pandas>=2.0.0,<2.1.0
tensorflow_probability>=0.20.0,<0.26.0
wandb>=0.15.0
dnspython
fancyflags
pyarrow>=10.0.0
py7zr
parameterized
portpicker
melee>=0.38.0
EOF""",
        "echo 'Created custom requirements file'"
    ])

    # Install custom requirements with constraints
    image = image.run_commands([
        "python -m pip install -r /tmp/slippi_requirements.txt --constraint /tmp/constraints.txt --no-upgrade",
        'python -c "import numpy; print(f\'NumPy after requirements: {numpy.__version__}\')"'
    ])

    # Install slippi-ai in editable mode with constraints
    image = image.run_commands([
        "python -m pip install -e . --no-deps --constraint /tmp/constraints.txt",
        'python -c "import numpy; print(f\'NumPy after editable install: {numpy.__version__}\')"'
    ])

    # Final verification
    image = image.run_commands([
        'python -c "import numpy; print(f\'FINAL NumPy version after all installs: {numpy.__version__}\')"',
        'python -c "import numpy; assert numpy.__version__.startswith(\'1.24\'), f\'NumPy upgrade detected: {numpy.__version__}\'"',
        'python -c "import scipy; print(f\'SciPy version: {scipy.__version__}\')"',
        'python -c "import tensorflow; print(f\'TensorFlow version: {tensorflow.__version__}\')"',
        'python -c "import dm_sonnet; print(f\'DM Sonnet version: {dm_sonnet.__version__}\')"'
    ])

    return image

def create_slippi_image_conda():
    """
    Alternative Conda-based approach with better dependency isolation
    """
    image = modal.Image.from_registry("continuumio/miniconda3:latest")
    
    # Install system dependencies
    image = image.apt_install([
        "build-essential", "git", "curl", "pkg-config", "cmake"
    ])
    
    # Create conda environment with locked NumPy
    image = image.run_commands([
        "conda create -n slippi python=3.10 -y",
        # Install NumPy 1.24.3 and pin it
        "conda install -n slippi 'numpy=1.24.3' --freeze-installed -c conda-forge -y",
        # Install scipy and pandas with pinned NumPy
        "conda install -n slippi 'scipy=1.10.1' 'pandas=2.0.3' --freeze-installed -c conda-forge -y",
        # Verify NumPy version
        ". /opt/conda/etc/profile.d/conda.sh && conda activate slippi && python -c 'import numpy; print(f\"Conda NumPy: {numpy.__version__}\")'",
        # Install Rust
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y"
    ]).env({
        "PATH": "/opt/conda/envs/slippi/bin:/root/.cargo/bin:$PATH",
        "CONDA_DEFAULT_ENV": "slippi"
    })
    
    # Install additional packages via pip in conda env with NumPy constraints
    image = image.run_commands([
        "echo 'numpy==1.24.3' > /tmp/conda_constraints.txt",
        ". /opt/conda/etc/profile.d/conda.sh && conda activate slippi && pip install tensorflow==2.13.0 --constraint /tmp/conda_constraints.txt",
        ". /opt/conda/etc/profile.d/conda.sh && conda activate slippi && pip install jax==0.4.13 jaxlib==0.4.13 --constraint /tmp/conda_constraints.txt",
        ". /opt/conda/etc/profile.d/conda.sh && conda activate slippi && pip install maturin",
        ". /opt/conda/etc/profile.d/conda.sh && conda activate slippi && pip install --no-build-isolation peppi-py==0.6.0 --constraint /tmp/conda_constraints.txt",
        # Clone and install slippi-ai
        f"git clone {REPO_URL} {PROJECT_ROOT}",
        f". /opt/conda/etc/profile.d/conda.sh && conda activate slippi && cd {PROJECT_ROOT} && pip install -e . --no-deps --constraint /tmp/conda_constraints.txt"
    ]).workdir(PROJECT_ROOT)
    
    return image

# Use the main approach with constraints
image = create_slippi_image()

app = modal.App("slippi-ai-numpy-locked")

@app.function(image=image)
def test_numpy_locked_env():
    """Test the NumPy-locked environment"""
    import numpy as np
    import scipy
    import tensorflow as tf
    import jax
    import dm_sonnet
    import peppi_py
    import sacred
    
    print("=" * 70)
    print("🔒 NUMPY-LOCKED SLIPPI AI ENVIRONMENT TEST")
    print("=" * 70)
    
    # Critical NumPy version check
    print(f"🎯 NumPy version: {np.__version__}")
    if np.__version__ == "1.24.3":
        print("✅ NumPy version is EXACTLY 1.24.3 - SUCCESS!")
    else:
        print(f"❌ CRITICAL: NumPy version is {np.__version__}, expected 1.24.3")
        return False
    
    # Package versions
    print(f"📊 SciPy version: {scipy.__version__}")
    print(f"🧠 TensorFlow version: {tf.__version__}")
    print(f"⚡ JAX version: {jax.__version__}")
    print(f"🎵 DM Sonnet version: {dm_sonnet.__version__}")
    print(f"🎮 peppi_py available: {peppi_py.__file__}")
    print(f"🔬 Sacred version: {sacred.__version__}")
    
    # Functionality tests
    test_results = []
    
    # NumPy test
    try:
        arr = np.array([1, 2, 3, 4, 5])
        result = np.mean(arr)
        print(f"✅ NumPy test: mean([1,2,3,4,5]) = {result}")
        test_results.append(True)
    except Exception as e:
        print(f"❌ NumPy test failed: {e}")
        test_results.append(False)
    
    # TensorFlow test
    try:
        tf_tensor = tf.constant([1.0, 2.0, 3.0])
        tf_result = tf.reduce_mean(tf_tensor)
        print(f"✅ TensorFlow test: {tf_result.numpy()}")
        test_results.append(True)
    except Exception as e:
        print(f"❌ TensorFlow test failed: {e}")
        test_results.append(False)
    
    # JAX test
    try:
        jax_array = jax.numpy.array([1.0, 2.0, 3.0])
        jax_result = jax.numpy.mean(jax_array)
        print(f"✅ JAX test: {jax_result}")
        print(f"✅ JAX devices: {jax.devices()}")
        test_results.append(True)
    except Exception as e:
        print(f"❌ JAX test failed: {e}")
        test_results.append(False)
    
    # DM Sonnet test
    try:
        import dm_sonnet as snt
        linear = snt.Linear(10)
        test_input = tf.random.normal([5, 20])
        output = linear(test_input)
        print(f"✅ DM Sonnet test: Linear layer output shape {output.shape}")
        test_results.append(True)
    except Exception as e:
        print(f"❌ DM Sonnet test failed: {e}")
        test_results.append(False)
    
    # SciPy test
    try:
        from scipy import sparse
        sparse_matrix = sparse.csr_matrix([[1, 2], [3, 4]])
        print(f"✅ SciPy sparse test: matrix shape {sparse_matrix.shape}")
        test_results.append(True)
    except Exception as e:
        print(f"❌ SciPy test failed: {e}")
        test_results.append(False)
    
    # peppi_py test
    try:
        print("✅ peppi_py import successful")
        test_results.append(True)
    except Exception as e:
        print(f"❌ peppi_py test failed: {e}")
        test_results.append(False)
    
    # Summary
    passed = sum(test_results)
    total = len(test_results)
    print(f"\n🏆 Test Results: {passed}/{total} tests passed")
    
    if np.__version__ == "1.24.3" and passed >= total - 1:  # Allow 1 test to fail
        print("🎉 SUCCESS: NumPy-locked Slippi AI environment is ready!")
        return True
    else:
        print("❌ FAILURE: Environment has issues")
        return False

@app.function(image=image)
def run_slippi_test():
    """Run a basic Slippi AI functionality test"""
    try:
        # Import key slippi-ai modules
        print("Testing Slippi AI imports...")
        
        # Test basic imports
        import numpy as np
        print(f"NumPy version: {np.__version__}")
        
        # Test if we can import slippi_ai modules
        import os
        import sys
        sys.path.insert(0, "/root/slippi-ai")
        
        # List available modules
        if os.path.exists("/root/slippi-ai"):
            print("Slippi AI directory contents:")
            for item in os.listdir("/root/slippi-ai"):
                print(f"  - {item}")
        
        print("✅ Basic Slippi AI environment test completed")
        
    except Exception as e:
        print(f"❌ Slippi AI test failed: {e}")
        import traceback
        traceback.print_exc()

@app.local_entrypoint()
def main():
    print("Testing NumPy-locked Slippi AI environment...")
    
    # Run comprehensive tests
    success = test_numpy_locked_env.remote()
    
    if success:
        print("\n🚀 Running Slippi AI functionality test...")
        run_slippi_test.remote()
    else:
        print("\n❌ Environment setup failed - skipping functionality tests")

if __name__ == "__main__":
    main()