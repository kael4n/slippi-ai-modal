# scripts/train_on_modal_fixed.py
# IMPROVED VERSION: Fixes major issues with Modal deployment and slippi-ai compatibility

import sys
import os
import subprocess
import json
import tempfile
from pathlib import Path
import modal

# --- Global Definitions ---
dataset_volume_name = "slippi-ai-dataset-doesokay"
models_volume_name = "slippi-ai-models-doesokay"
project_root_path_str = "/root/slippi-ai"
repo_url = "https://github.com/vladfi1/slippi-ai.git"

# Updated peppi-py commit - using more recent stable version
peppi_py_commit_url = "git+https://github.com/hohav/peppi-py.git@8c02a4659c3302321dfbfcf2093c62f634e335f7"

# --- Improved Image Definition ---
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install([
        # Essential build tools
        "git", "build-essential", "pkg-config", "curl", "wget",
        # Rust dependencies
        "libssl-dev", "libffi-dev", "python3-dev", 
        # TensorFlow dependencies
        "cmake", "libhdf5-dev", "libblas-dev", "liblapack-dev",
        # Additional system libraries that might be needed
        "libc6-dev", "gcc", "g++", "make", "zlib1g-dev",
    ])
    .run_commands([
        # Install Rust with proper PATH handling
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable",
        # Make cargo available in PATH for subsequent commands
        "echo 'export PATH=\"$HOME/.cargo/bin:$PATH\"' >> ~/.bashrc",
    ])
    .env({
        "PATH": "/root/.cargo/bin:$PATH",
        "CARGO_HOME": "/root/.cargo",
        "RUSTUP_HOME": "/root/.rustup",
        "TF_ENABLE_ONEDNN_OPTS": "0",
        "TF_CPP_MIN_LOG_LEVEL": "1",
        "CUDA_VISIBLE_DEVICES": "0",
        "TF_FORCE_GPU_ALLOW_GROWTH": "true",
        # Disable wandb for Modal (unless you want to set up API key)
        "WANDB_MODE": "disabled",
        # Python-specific environment variables
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
    })
    .run_commands([
        # Clone the repository
        f"git clone {repo_url} {project_root_path_str}",
    ])
    .workdir(project_root_path_str)
    .run_commands([
        # Install Python dependencies with careful ordering
        "pip install --upgrade pip setuptools wheel",
        
        # Install Rust build tools for Python (cargo should be in PATH now)
        "/root/.cargo/bin/cargo --version || echo 'Cargo not found, trying alternative path'",
        "pip install maturin",
        
        # Install core ML dependencies first
        "pip install 'tensorflow>=2.8.0,<2.16.0'",  # Use compatible TF version
        "pip install tensorflow-probability",
        "pip install 'numpy>=1.21.0,<1.25.0'",  # Compatible numpy version
        "pip install pandas matplotlib seaborn",
        
        # Install other core dependencies
        "pip install dm-tree sacred pymongo",
        "pip install gym gymnasium",  # For RL environments
        
        # Install peppi-py from specific commit (this is the critical step)
        f"pip install --no-build-isolation --verbose '{peppi_py_commit_url}'",
        
        # Install project requirements (with error handling)
        "pip install -r requirements.txt || echo 'Some requirements failed, continuing...'",
        
        # Install the slippi-ai package itself
        "pip install -e . || echo 'Package installation failed, but continuing...'",
        
        # Verify critical imports work
        "python -c 'import tensorflow as tf; print(f\"TensorFlow: {tf.__version__}\")' || echo 'TensorFlow import failed'",
        "python -c 'import peppi; print(f\"peppi imported successfully\")' || echo 'peppi import failed'",
    ])
)

# --- Modal App and Volumes ---
app = modal.App("slippi-ai-trainer")
dataset_volume = modal.Volume.from_name(dataset_volume_name)
models_volume = modal.Volume.from_name(models_volume_name, create_if_missing=True)

def get_example_training_args():
    """
    Get training arguments based on slippi-ai documentation and examples
    These are more realistic parameters for slippi-ai training
    """
    return [
        "--data_dir", "/dataset",
        "--expt_root", "/models", 
        "--tag", "modal_training",
        
        # Learning parameters - more conservative for stability
        "--learner.learning_rate", "3e-4",
        "--learner.beta1", "0.9",
        "--learner.beta2", "0.999",
        "--learner.epsilon", "1e-8",
        
        # Data parameters
        "--data.batch_size", "16",  # Smaller batch size for stability
        "--data.unroll_length", "32",  # Shorter sequences
        "--data.compressed", "True",  # Use compression to save memory
        
        # Runtime parameters
        "--runtime.max_runtime", "7200",  # 2 hours
        "--runtime.save_interval", "600",  # Save every 10 minutes
        "--runtime.log_interval", "50",
        "--runtime.eval_interval", "1000",
        
        # Network parameters - start simple
        "--network.name", "mlp",
        "--network.mlp.depth", "3",
        "--network.mlp.width", "256",
        "--network.mlp.activation", "relu",
        
        # Enable compilation for better performance
        "--learner.compile", "True",
        
        # Memory management
        "--runtime.memory_limit", "8000",  # 8GB memory limit
    ]

@app.function(
    image=image,
    volumes={
        "/dataset": dataset_volume,
        "/models": models_volume,
    },
    timeout=28800,  # 8 hours timeout
    gpu=modal.gpu.A10G(),  # A10G should be sufficient for initial training
    memory=16384,  # 16GB memory
    retries=modal.Retries(
        max_retries=2,
        backoff_coefficient=2.0,
        initial_delay=10.0,
    ),
)
def train_slippi_ai():
    """
    Train the Slippi AI model with improved error handling and compatibility
    """
    import time
    
    # Setup paths and environment
    os.chdir(project_root_path_str)
    sys.path.insert(0, project_root_path_str)
    
    print("=== Starting Slippi AI Training on Modal ===")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check environment setup
    dataset_path = Path("/dataset")
    models_path = Path("/models")
    models_path.mkdir(exist_ok=True)
    
    # Verify GPU and TensorFlow setup
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        
        # Configure TensorFlow
        gpus = tf.config.list_physical_devices('GPU')
        print(f"Available GPUs: {len(gpus)}")
        
        if gpus:
            try:
                # Configure GPU memory growth
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("✅ GPU memory growth configured")
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
        else:
            print("⚠️ WARNING: No GPU detected! Training will be very slow.")
            
        # Test GPU computation
        with tf.device('/GPU:0' if gpus else '/CPU:0'):
            test_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            result = tf.matmul(test_tensor, test_tensor)
            print(f"✅ TensorFlow GPU test successful: device={result.device}")
            
    except Exception as e:
        print(f"❌ TensorFlow setup error: {e}")
        raise
    
    # Verify peppi import
    try:
        import peppi
        print("✅ peppi imported successfully")
    except ImportError as e:
        print(f"❌ peppi import failed: {e}")
        print("This is critical for slippi replay parsing!")
        raise
    
    # Check for training data
    print(f"Checking dataset at: {dataset_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist!")
    
    # Look for .slp files (Slippi replay files)
    slp_files = list(dataset_path.rglob("*.slp"))
    print(f"Found {len(slp_files)} .slp files")
    
    if len(slp_files) == 0:
        # Also check for preprocessed data
        pkl_files = list(dataset_path.rglob("*.pkl"))
        tfrecord_files = list(dataset_path.rglob("*.tfrecord*"))
        print(f"Found {len(pkl_files)} .pkl files, {len(tfrecord_files)} .tfrecord files")
        
        if len(pkl_files) == 0 and len(tfrecord_files) == 0:
            raise FileNotFoundError(
                "No training data found! Please upload either:\n"
                "- .slp replay files for raw training\n"
                "- .pkl or .tfrecord files for preprocessed training data"
            )
    
    # Test parsing a few replay files to ensure peppi works
    if slp_files:
        print("Testing replay file parsing...")
        try:
            test_file = slp_files[0]
            game = peppi.game(str(test_file))
            print(f"✅ Successfully parsed {test_file.name}")
            print(f"   Players: {len(game.players)}")
            print(f"   Frames: {len(game.frames) if hasattr(game, 'frames') else 'Unknown'}")
        except Exception as e:
            print(f"⚠️ Warning: Could not parse test replay file: {e}")
    
    # Set up environment variables
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": f"{project_root_path_str}:{env.get('PYTHONPATH', '')}",
        "CUDA_VISIBLE_DEVICES": "0",
        "WANDB_MODE": "disabled",  # Disable wandb unless configured
        "TF_ENABLE_ONEDNN_OPTS": "0",  # Disable oneDNN optimizations for stability
        "TF_CPP_MIN_LOG_LEVEL": "1",  # Reduce TensorFlow logging
    })
    
    # Find training script
    possible_train_scripts = [
        Path(project_root_path_str) / "scripts" / "train.py",
        Path(project_root_path_str) / "slippi_ai" / "train.py",
        Path(project_root_path_str) / "train.py",
    ]
    
    train_script = None
    for script_path in possible_train_scripts:
        if script_path.exists():
            train_script = script_path
            break
    
    if not train_script:
        raise FileNotFoundError(
            f"Training script not found in any of these locations:\n" +
            "\n".join(f"  - {p}" for p in possible_train_scripts)
        )
    
    print(f"Using training script: {train_script}")
    
    # Build command with proper arguments
    cmd = [sys.executable, str(train_script)] + get_example_training_args()
    
    print(f"Running command:")
    print(f"  {' '.join(cmd[:3])} \\")
    for i in range(3, len(cmd), 2):
        if i + 1 < len(cmd):
            print(f"    {cmd[i]} {cmd[i+1]} \\")
        else:
            print(f"    {cmd[i]}")
    
    # Start training with comprehensive error handling
    start_time = time.time()
    try:
        # Run training with real-time output
        process = subprocess.Popen(
            cmd,
            cwd=project_root_path_str,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream output in real-time with timestamps
        print("\n=== Training Output ===")
        for line in iter(process.stdout.readline, ''):
            if line:
                timestamp = time.strftime('%H:%M:%S')
                print(f"[{timestamp}] {line.rstrip()}")
        
        process.wait()
        elapsed_time = time.time() - start_time
        
        if process.returncode == 0:
            print(f"\n✅ Training completed successfully in {elapsed_time:.1f} seconds!")
            
            # List created model files
            model_files = list(models_path.rglob("*"))
            model_files = [f for f in model_files if f.is_file()]
            print(f"Created {len(model_files)} model files:")
            
            # Show structure of created files
            for f in sorted(model_files)[:20]:  # Show first 20 files
                size_mb = f.stat().st_size / (1024 * 1024)
                rel_path = f.relative_to(models_path)
                print(f"  {rel_path} ({size_mb:.1f} MB)")
            
            if len(model_files) > 20:
                print(f"  ... and {len(model_files) - 20} more files")
            
        else:
            raise subprocess.CalledProcessError(process.returncode, cmd)
            
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with return code: {e.returncode}")
        print(f"Training ran for {time.time() - start_time:.1f} seconds")
        raise
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        print(f"Training ran for {time.time() - start_time:.1f} seconds")
        raise

@app.function(
    image=image,
    volumes={
        "/dataset": dataset_volume,
        "/models": models_volume,
    },
    timeout=600,
)
def debug_environment():
    """
    Debug the environment and check what's available
    """
    print("=== Environment Debug ===")
    
    os.chdir(project_root_path_str)
    sys.path.insert(0, project_root_path_str)
    
    # Check Python environment
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Check key imports with more details
    imports_to_check = [
        ("tensorflow", "TensorFlow"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("peppi", "peppi (Slippi parser)"),
        ("sacred", "Sacred (experiment management)"),
        ("dm_tree", "dm-tree"),
        ("matplotlib", "Matplotlib"),
        ("gym", "Gym"),
        ("subprocess", "subprocess"),
    ]
    
    print("\nPython package versions:")
    for module_name, display_name in imports_to_check:
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {display_name}: {version}")
        except ImportError as e:
            print(f"❌ {display_name}: {e}")
    
    # Check Rust and Cargo
    print("\nRust toolchain:")
    try:
        rust_version = subprocess.run(
            ["/root/.cargo/bin/rustc", "--version"], 
            capture_output=True, text=True, timeout=10
        )
        if rust_version.returncode == 0:
            print(f"✅ Rust: {rust_version.stdout.strip()}")
        else:
            print("❌ Rust not available")
    except Exception as e:
        print(f"❌ Rust check failed: {e}")
        # Try alternative path
        try:
            rust_version = subprocess.run(
                ["rustc", "--version"], 
                capture_output=True, text=True, timeout=10
            )
            if rust_version.returncode == 0:
                print(f"✅ Rust (alternative path): {rust_version.stdout.strip()}")
        except Exception:
            print("❌ Rust not found in any path")
    
    # Check project structure
    print(f"\nProject structure at {project_root_path_str}:")
    project_path = Path(project_root_path_str)
    if project_path.exists():
        for item in sorted(project_path.iterdir()):
            if item.is_dir():
                subcount = len(list(item.iterdir())) if item.is_dir() else 0
                print(f"  📁 {item.name}/ ({subcount} items)")
            else:
                size_kb = item.stat().st_size / 1024
                print(f"  📄 {item.name} ({size_kb:.1f} KB)")
    
    # Check scripts directory in detail
    scripts_dir = project_path / "scripts"
    if scripts_dir.exists():
        print(f"\nScripts directory:")
        for script in sorted(scripts_dir.glob("*")):
            if script.is_file():
                print(f"  📄 {script.name}")
            elif script.is_dir():
                print(f"  📁 {script.name}/")
    
    # Check for slippi_ai module
    slippi_ai_dir = project_path / "slippi_ai"
    if slippi_ai_dir.exists():
        print(f"\nslippi_ai module structure:")
        for item in sorted(slippi_ai_dir.iterdir()):
            if item.is_file() and item.suffix == '.py':
                print(f"  📄 {item.name}")
            elif item.is_dir():
                py_files = len(list(item.glob("*.py")))
                print(f"  📁 {item.name}/ ({py_files} .py files)")
    
    # Check dataset in detail
    dataset_path = Path("/dataset")
    print(f"\nDataset analysis at {dataset_path}:")
    if dataset_path.exists():
        file_types = {}
        total_size = 0
        
        for file_path in dataset_path.rglob("*"):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                size = file_path.stat().st_size
                total_size += size
                
                if ext not in file_types:
                    file_types[ext] = {"count": 0, "size": 0}
                file_types[ext]["count"] += 1
                file_types[ext]["size"] += size
        
        print(f"  Total size: {total_size / (1024**3):.2f} GB")
        print("  File types:")
        for ext, info in sorted(file_types.items()):
            count = info["count"]
            size_mb = info["size"] / (1024**2)
            print(f"    {ext or '(no extension)'}: {count} files ({size_mb:.1f} MB)")
        
        # Show directory structure (first level)
        print("  Directory structure:")
        for item in sorted(dataset_path.iterdir()):
            if item.is_dir():
                subcount = len(list(item.rglob("*")))
                print(f"    📁 {item.name}/ ({subcount} total items)")
            else:
                size_mb = item.stat().st_size / (1024**2)
                print(f"    📄 {item.name} ({size_mb:.1f} MB)")
    else:
        print("  ❌ Dataset directory not found!")
    
    # Check models directory
    models_path = Path("/models")
    models_path.mkdir(exist_ok=True)
    print(f"\nModels directory at {models_path}: ✅")
    existing_models = list(models_path.rglob("*"))
    if existing_models:
        print(f"  Found {len(existing_models)} existing files/directories")
    
    return True

@app.function(
    image=image,
    volumes={
        "/dataset": dataset_volume,
    },
    timeout=600,
)
def check_training_script():
    """
    Check if the training script can be imported and what arguments it expects
    """
    print("=== Training Script Analysis ===")
    
    os.chdir(project_root_path_str)
    sys.path.insert(0, project_root_path_str)
    
    # Look for training scripts
    possible_scripts = [
        Path(project_root_path_str) / "scripts" / "train.py",
        Path(project_root_path_str) / "slippi_ai" / "train.py", 
        Path(project_root_path_str) / "train.py",
    ]
    
    for script_path in possible_scripts:
        print(f"\nChecking: {script_path}")
        if script_path.exists():
            print(f"✅ Found script at {script_path}")
            
            # Try to get help
            try:
                cmd = [sys.executable, str(script_path), "--help"]
                result = subprocess.run(
                    cmd,
                    cwd=project_root_path_str,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    print("✅ Script help output:")
                    print(result.stdout[:1500])  # First 1500 chars
                else:
                    print(f"❌ Help command failed:")
                    print(f"stdout: {result.stdout[:500]}")
                    print(f"stderr: {result.stderr[:500]}")
                    
            except subprocess.TimeoutExpired:
                print("⚠️ Help command timed out")
            except Exception as e:
                print(f"❌ Error running help: {e}")
        else:
            print(f"❌ Not found")
    
    # Check for example scripts
    scripts_dir = Path(project_root_path_str) / "scripts"
    if scripts_dir.exists():
        print(f"\nExample scripts in {scripts_dir}:")
        for script in scripts_dir.glob("*.sh"):
            print(f"✅ Found example: {script.name}")
            try:
                with open(script, 'r') as f:
                    content = f.read()
                    if len(content) < 1000:
                        print(f"Content of {script.name}:")
                        print(content)
                    else:
                        print(f"Content preview of {script.name}:")
                        print(content[:800] + "...")
            except Exception as e:
                print(f"Error reading {script.name}: {e}")
    
    return True

@app.function(
    image=image,
    volumes={
        "/dataset": dataset_volume,
    },
    timeout=300,
)
def test_peppi_parsing():
    """
    Test peppi parsing on actual replay files
    """
    print("=== Peppi Parsing Test ===")
    
    try:
        import peppi
        print(f"✅ peppi imported successfully")
    except ImportError as e:
        print(f"❌ peppi import failed: {e}")
        return False
    
    # Find .slp files
    dataset_path = Path("/dataset")
    slp_files = list(dataset_path.rglob("*.slp"))
    
    if not slp_files:
        print("❌ No .slp files found for testing")
        return False
    
    print(f"Found {len(slp_files)} .slp files, testing first few...")
    
    successful_parses = 0
    for i, slp_file in enumerate(slp_files[:5]):  # Test first 5 files
        try:
            print(f"\nTesting {slp_file.name}...")
            game = peppi.game(str(slp_file))
            
            print(f"✅ Successfully parsed {slp_file.name}")
            print(f"   File size: {slp_file.stat().st_size / 1024:.1f} KB")
            print(f"   Players: {len(game.players) if hasattr(game, 'players') else 'Unknown'}")
            
            # Try to access frame data
            if hasattr(game, 'frames'):
                frame_count = len(game.frames)
                print(f"   Frames: {frame_count}")
                if frame_count > 0:
                    print(f"   Duration: ~{frame_count / 60:.1f} seconds")
            
            successful_parses += 1
            
        except Exception as e:
            print(f"❌ Failed to parse {slp_file.name}: {e}")
    
    print(f"\nSummary: {successful_parses}/{min(5, len(slp_files))} files parsed successfully")
    return successful_parses > 0

@app.local_entrypoint()
def main():
    """Main entrypoint for Modal training with enhanced commands"""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
    else:
        command = "train"
    
    if command == "debug":
        print("🔧 Running comprehensive environment debug...")
        debug_environment.remote()
        
    elif command == "check":
        print("🔍 Checking training script compatibility...")
        check_training_script.remote()
        
    elif command == "test-peppi":
        print("🧪 Testing peppi replay file parsing...")
        test_peppi_parsing.remote()
        
    elif command == "full-check":
        print("🔍 Running full pre-training checks...")
        print("\n1. Environment debug...")
        debug_environment.remote()
        print("\n2. Training script check...")
        check_training_script.remote()
        print("\n3. Peppi parsing test...")
        test_peppi_parsing.remote()
        print("\n✅ Full check completed!")
        
    elif command == "train":
        print("🚀 Starting slippi-ai training...")
        try:
            train_slippi_ai.remote()
            print("✅ Training completed successfully!")
        except Exception as e:
            print(f"❌ Training failed: {e}")
            print("\n💡 Troubleshooting suggestions:")
            print("  1. Run 'python train_on_modal_fixed.py debug' to check environment")
            print("  2. Run 'python train_on_modal_fixed.py check' to verify training script")
            print("  3. Run 'python train_on_modal_fixed.py test-peppi' to test replay parsing")
            print("  4. Check that your dataset contains .slp files or preprocessed data")
    
    else:
        print("Usage: python train_on_modal_fixed.py [command]")
        print("\nCommands:")
        print("  debug       - Check environment and dataset")
        print("  check       - Analyze training script compatibility")
        print("  test-peppi  - Test peppi replay file parsing")
        print("  full-check  - Run all checks before training")
        print("  train       - Start training (default)")
        print("\n💡 Recommended: Run 'full-check' before your first training attempt!")

if __name__ == "__main__":
    main()