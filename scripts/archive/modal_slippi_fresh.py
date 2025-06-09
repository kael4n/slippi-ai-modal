# modal_slippi_fresh.py - Fixed version with proper Modal CLI
import modal
import os
import sys
import subprocess
from pathlib import Path

# --- Configuration ---
dataset_volume_name = "slippi-ai-dataset-doesokay"
models_volume_name = "slippi-ai-models-doesokay"
project_root_path_str = "/root/slippi-ai"
repo_url = "https://github.com/vladfi1/slippi-ai.git"

# --- Build image step by step with proper maturin installation ---
image = (
    modal.Image.from_registry(
        "tensorflow/tensorflow:2.12.0-gpu"
        # Don't add_python - use the existing Python from TensorFlow image
    )
    .apt_install([
        "git", "build-essential", "pkg-config", "curl", "wget",
        "libssl-dev", "libffi-dev", "python3-dev", "cmake",
        "libc6-dev", "gcc", "g++", "make", "zlib1g-dev"
    ])
    # Step 1: Upgrade pip and install basic build tools
    .run_commands([
        "python -m pip install --upgrade pip setuptools wheel"
    ])
    
    # Step 1.5: Ensure TensorFlow is available
    .run_commands([
        "python -c 'import tensorflow as tf; print(f\"TensorFlow version: {tf.__version__}\")'",
    ])
    
    # Step 2: Install Rust
    .run_commands([
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
    ])
    .env({"PATH": "/root/.cargo/bin:$PATH"})
    
    # Step 3: Install maturin (CRITICAL - must be after Rust installation)
    .pip_install(["maturin"])
    
    # Step 4: Clone repository
    .run_commands([
        f"git clone {repo_url} {project_root_path_str}",
    ])
    .workdir(project_root_path_str)
    
    # Step 5: Install peppi-py with explicit flags (now maturin is available)
    .run_commands([
        "pip install --no-build-isolation --verbose 'peppi-py @ git+https://github.com/hohav/peppi-py.git@8c02a4659c3302321dfbfcf2093c62f634e335f7'"
    ])
    
    # Step 6: Install other Python dependencies
    .pip_install([
        "sacred", "pymongo", "pandas", "matplotlib", "seaborn",
        "dm-tree", "gym", "gymnasium"
    ])
    
    # Step 7: Install project requirements
    .run_commands([
        "pip install -r requirements.txt || echo 'Some requirements failed'",
        "pip install -e . || echo 'Package installation failed'"
    ])
)

# --- Modal App ---
app = modal.App("slippi-ai-fresh")  # Different app name to avoid caching
dataset_volume = modal.Volume.from_name(dataset_volume_name)
models_volume = modal.Volume.from_name(models_volume_name, create_if_missing=True)

@app.function(
    image=image,
    timeout=300
)
def test_environment():
    """Test the environment setup"""
    print("=== Environment Test ===")
    
    try:
        # Test imports
        import tensorflow as tf
        print(f"✅ TensorFlow: {tf.__version__}")
        
        import peppi
        print("✅ peppi imported successfully")
        
        import sacred
        print("✅ sacred imported")
        
        # Test GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        print(f"✅ GPUs available: {len(gpus)}")
        
        # Test simple computation
        x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        y = tf.matmul(x, x)
        print(f"✅ TensorFlow computation test: {y.shape}")
        
        return "✅ All tests passed!"
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise

@app.function(
    image=image,
    volumes={
        "/dataset": dataset_volume,
        "/models": models_volume,
    },
    timeout=28800,
    gpu="A10G",
    memory=16384,
    retries=modal.Retries(max_retries=2),
)
def train_model():
    """Train the Slippi AI model"""
    import time
    
    os.chdir(project_root_path_str)
    sys.path.insert(0, project_root_path_str)
    
    print("=== Slippi AI Training ===")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Environment check
    try:
        import tensorflow as tf
        import peppi
        print(f"✅ TensorFlow: {tf.__version__}")
        print(f"✅ peppi available")
        
        # GPU setup
        gpus = tf.config.list_physical_devices('GPU')
        print(f"✅ GPUs available: {len(gpus)}")
        
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                
    except Exception as e:
        print(f"❌ Environment check failed: {e}")
        raise
    
    # Check dataset
    dataset_path = Path("/dataset")
    if not dataset_path.exists():
        raise FileNotFoundError("Dataset not found at /dataset!")
    
    slp_files = list(dataset_path.rglob("*.slp"))
    print(f"Found {len(slp_files)} .slp files in dataset")
    
    if len(slp_files) == 0:
        raise FileNotFoundError("No .slp files found in dataset!")
    
    # Training configuration
    training_args = [
        "--data_dir", "/dataset",
        "--expt_root", "/models", 
        "--tag", "fresh_training",
        "--learner.learning_rate", "3e-4",
        "--data.batch_size", "16",
        "--data.unroll_length", "32",
        "--runtime.max_runtime", "7200",  # 2 hours
        "--runtime.save_interval", "600",  # Save every 10 minutes
        "--runtime.log_interval", "50",
        "--network.name", "mlp",
        "--network.mlp.depth", "3",
        "--network.mlp.width", "256",
        "--learner.compile", "True",
    ]
    
    # Find training script
    train_script = None
    possible_scripts = [
        Path(project_root_path_str) / "scripts" / "train.py",
        Path(project_root_path_str) / "slippi_ai" / "train.py", 
        Path(project_root_path_str) / "train.py",
    ]
    
    for script_path in possible_scripts:
        if script_path.exists():
            train_script = script_path
            print(f"Found training script: {script_path}")
            break
    
    if not train_script:
        print("Available files in project root:")
        for f in Path(project_root_path_str).rglob("*.py"):
            print(f"  {f}")
        raise FileNotFoundError("Training script not found!")
    
    # Execute training
    cmd = [sys.executable, str(train_script)] + training_args
    print(f"Running command: {' '.join(cmd[:6])}...")
    
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": "0",
        "WANDB_MODE": "disabled",
        "TF_ENABLE_ONEDNN_OPTS": "0",
        "TF_CPP_MIN_LOG_LEVEL": "1",
    })
    
    start_time = time.time()
    try:
        process = subprocess.Popen(
            cmd,
            cwd=project_root_path_str,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream output
        for line in iter(process.stdout.readline, ''):
            if line:
                timestamp = time.strftime('%H:%M:%S')
                print(f"[{timestamp}] {line.rstrip()}")
        
        process.wait()
        elapsed = time.time() - start_time
        
        if process.returncode == 0:
            print(f"✅ Training completed successfully in {elapsed:.1f}s")
        else:
            raise subprocess.CalledProcessError(process.returncode, cmd)
            
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise

# Separate entrypoints for each command
@app.local_entrypoint()
def test():
    """Test the environment setup"""
    print("🧪 Testing environment...")
    result = test_environment.remote()
    print(f"Test result: {result}")

@app.local_entrypoint()
def train():
    """Run model training"""
    print("🚀 Starting training...")
    train_model.remote()
    print("✅ Training job submitted!")

# Keep the old main for backward compatibility
@app.local_entrypoint()
def main():
    """Default entrypoint - runs training"""
    print("🚀 Starting training (default)...")
    train_model.remote()
    print("✅ Training job submitted!")

if __name__ == "__main__":
    main()