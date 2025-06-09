# modal_docker_slippi.py
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

# --- Option 1: Use existing TensorFlow image ---
image_from_registry = (
    modal.Image.from_registry(
        "tensorflow/tensorflow:2.12.0-gpu",
        add_python="3.10"
    )
    .apt_install([
        "git", "build-essential", "pkg-config", "curl", "wget",
        "libssl-dev", "libffi-dev", "python3-dev", "cmake"
    ])
    .run_commands([
        # Install Rust
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        "echo 'export PATH=\"$HOME/.cargo/bin:$PATH\"' >> ~/.bashrc",
    ])
    .env({"PATH": "/root/.cargo/bin:$PATH"})
    .run_commands([
        f"git clone {repo_url} {project_root_path_str}",
    ])
    .workdir(project_root_path_str)
    .pip_install([
        "peppi-py @ git+https://github.com/hohav/peppi-py.git@8c02a4659c3302321dfbfcf2093c62f634e335f7",
        "sacred", "pymongo", "pandas", "matplotlib", "seaborn",
        "dm-tree", "gym", "gymnasium"
    ])
    .run_commands([
        "pip install -r requirements.txt || echo 'Some requirements failed'",
        "pip install -e . || echo 'Package installation failed'"
    ])
)

# --- Option 2: Use custom Dockerfile ---
# Create a Dockerfile that you can reuse
dockerfile_content = '''
FROM tensorflow/tensorflow:2.12.0-gpu

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    git build-essential pkg-config curl wget \\
    libssl-dev libffi-dev python3-dev cmake \\
    libc6-dev gcc g++ make zlib1g-dev

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"

# Clone slippi-ai repo
RUN git clone https://github.com/vladfi1/slippi-ai.git /root/slippi-ai
WORKDIR /root/slippi-ai

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install maturin

# Install peppi-py (this is the tricky one)
RUN pip install --no-build-isolation --verbose \\
    "peppi-py @ git+https://github.com/hohav/peppi-py.git@8c02a4659c3302321dfbfcf2093c62f634e335f7"

# Install other dependencies
RUN pip install sacred pymongo pandas matplotlib seaborn dm-tree gym gymnasium

# Install project requirements
RUN pip install -r requirements.txt || echo "Some requirements failed"
RUN pip install -e . || echo "Package installation failed"

# Set environment variables
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV TF_CPP_MIN_LOG_LEVEL=1
ENV WANDB_MODE=disabled
ENV PYTHONUNBUFFERED=1
'''

# Write Dockerfile to a temporary location
def create_dockerfile():
    dockerfile_path = Path("./Dockerfile.slippi")
    dockerfile_path.write_text(dockerfile_content)
    return dockerfile_path

# Use the Dockerfile
dockerfile_path = create_dockerfile()
image_from_dockerfile = modal.Image.from_dockerfile(dockerfile_path)

# --- Option 3: Use a pre-built image from Docker Hub ---
# You could build and push your own image to Docker Hub, then use it
# image_prebuild = modal.Image.from_registry("yourusername/slippi-ai:latest")

# Choose which image to use
image = image_from_registry  # or image_from_dockerfile

# --- Modal App ---
app = modal.App("slippi-ai-docker")
dataset_volume = modal.Volume.from_name(dataset_volume_name)
models_volume = modal.Volume.from_name(models_volume_name, create_if_missing=True)

@app.function(
    image=image,
    volumes={
        "/dataset": dataset_volume,
        "/models": models_volume,
    },
    timeout=28800,
    gpu=modal.gpu.A10G(),
    memory=16384,
    retries=modal.Retries(max_retries=2),
)
def train_slippi_ai():
    """Train with Docker-based image"""
    import time
    
    os.chdir(project_root_path_str)
    sys.path.insert(0, project_root_path_str)
    
    print("=== Training with Docker Image ===")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Quick environment check
    try:
        import tensorflow as tf
        import peppi
        print(f"✅ TensorFlow: {tf.__version__}")
        print(f"✅ peppi imported successfully")
        
        # Check GPU
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
        raise FileNotFoundError("Dataset not found!")
    
    slp_files = list(dataset_path.rglob("*.slp"))
    print(f"Found {len(slp_files)} .slp files")
    
    # Training arguments
    training_args = [
        "--data_dir", "/dataset",
        "--expt_root", "/models",
        "--tag", "docker_training",
        "--learner.learning_rate", "3e-4",
        "--data.batch_size", "16",
        "--data.unroll_length", "32",
        "--runtime.max_runtime", "7200",
        "--runtime.save_interval", "600",
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
            break
    
    if not train_script:
        raise FileNotFoundError("Training script not found!")
    
    # Run training
    cmd = [sys.executable, str(train_script)] + training_args
    print(f"Running: {' '.join(cmd[:5])} ...")
    
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": "0",
        "WANDB_MODE": "disabled",
        "TF_ENABLE_ONEDNN_OPTS": "0",
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
        
        for line in iter(process.stdout.readline, ''):
            if line:
                timestamp = time.strftime('%H:%M:%S')
                print(f"[{timestamp}] {line.rstrip()}")
        
        process.wait()
        elapsed = time.time() - start_time
        
        if process.returncode == 0:
            print(f"✅ Training completed in {elapsed:.1f}s")
        else:
            raise subprocess.CalledProcessError(process.returncode, cmd)
            
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise

@app.function(image=image, timeout=300)
def quick_test():
    """Quick test of the Docker image environment"""
    print("=== Docker Image Test ===")
    
    # Test critical imports
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow: {tf.__version__}")
        
        import peppi
        print("✅ peppi imported")
        
        import sacred
        print("✅ sacred imported")
        
        # Test GPU
        gpus = tf.config.list_physical_devices('GPU')
        print(f"✅ GPUs: {len(gpus)}")
        
        # Test simple computation
        with tf.device('/GPU:0' if gpus else '/CPU:0'):
            x = tf.random.normal([100, 100])
            y = tf.matmul(x, x)
            print(f"✅ Computation test: {y.shape}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
    
    return "Docker image test passed!"

@app.local_entrypoint()
def main():
    command = sys.argv[1] if len(sys.argv) > 1 else "train"
    
    if command == "test":
        print("🧪 Testing Docker image...")
        result = quick_test.remote()
        print(f"Result: {result}")
        
    elif command == "train":
        print("🚀 Starting training with Docker image...")
        train_slippi_ai.remote()
        print("✅ Training completed!")
        
    else:
        print("Usage: python modal_docker_slippi.py [test|train]")

if __name__ == "__main__":
    main()
