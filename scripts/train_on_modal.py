# scripts/train_on_modal.py
# FIXED VERSION: Properly handles the config-based training script interface

import sys
import os
import subprocess  # Move this to module level
import json
import tempfile
from pathlib import Path

import modal

# --- Global Definitions ---
dataset_volume_name = "slippi-ai-dataset-doesokay"
models_volume_name = "slippi-ai-models-doesokay"
project_root_path_str = "/root/slippi-ai"
repo_url = "https://github.com/vladfi1/slippi-ai.git"
# The specific commit for peppi-py required by slippi-ai
peppi_py_commit_url = "git+https://github.com/hohav/peppi-py.git@8c02a4659c3302321dfbfcf2093c62f634e335f7"

# --- Image Definition (Using Modal's recommended approach for GPU support) ---
image = (
    # Start with Modal's Python image which has better CUDA support
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "build-essential", "pkg-config", "libssl-dev", "curl")
    .run_commands([
        # Install Rust toolchain first
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        # Use . instead of source for sh compatibility
        ". ~/.cargo/env && cargo --version",
    ])
    .env({
        "PATH": "/root/.cargo/bin:$PATH",
        "CARGO_HOME": "/root/.cargo",
        "RUSTUP_HOME": "/root/.rustup",
        # TensorFlow environment variables
        "TF_ENABLE_ONEDNN_OPTS": "0",  # Disable oneDNN warnings
        "TF_CPP_MIN_LOG_LEVEL": "1",   # Reduce TF logging
        # CUDA environment variables
        "CUDA_VISIBLE_DEVICES": "0",
        "TF_FORCE_GPU_ALLOW_GROWTH": "true",
    })
    .run_commands([
        f"git clone {repo_url} {project_root_path_str}",
    ])
    .workdir(project_root_path_str)
    .run_commands([
        # Verify Rust is available and install Python dependencies
        ". ~/.cargo/env && cargo --version",
        "pip install --upgrade pip setuptools wheel",
        ". ~/.cargo/env && pip install maturin",
        "pip install -r requirements.txt",
        # Install peppi-py with Rust environment properly sourced
        f". ~/.cargo/env && pip install --no-build-isolation '{peppi_py_commit_url}'",
        # Install the slippi-ai package itself in development mode
        "pip install -e .",
    ])
)

# --- Modal App and Volumes ---
app = modal.App("slippi-ai-trainer")
dataset_volume = modal.Volume.from_name(dataset_volume_name)
models_volume = modal.Volume.from_name(models_volume_name, create_if_missing=True)

def create_training_config(dataset_path, models_path, experiment_name="modal_training"):
    """
    Create a proper configuration for the training script based on the default config
    """
    config = {
        'runtime': {
            'max_runtime': 3600,  # 1 hour
            'log_interval': 10,
            'save_interval': 300,  # Save every 5 minutes
            'eval_every_n': 100,
            'num_eval_steps': 10
        },
        'dataset': {
            'data_dir': str(dataset_path),
            'meta_path': None,
            'test_ratio': 0.1,
            'allowed_characters': 'all',
            'allowed_opponents': 'all',
            'allowed_names': 'all',
            'banned_names': 'none',
            'swap': True,
            'seed': 0
        },
        'data': {
            'batch_size': 32,
            'unroll_length': 64,
            'damage_ratio': 0.01,
            'compressed': True,
            'num_workers': 0
        },
        'learner': {
            'learning_rate': 0.0001,
            'compile': True,
            'jit_compile': True,
            'decay_rate': 0.0,
            'value_cost': 0.5,
            'reward_halflife': 4.0
        },
        'network': {
            'name': 'mlp',
            'mlp': {
                'depth': 2,
                'width': 128,
                'dropout_rate': 0.0
            },
            'lstm': {
                'hidden_size': 128,
                'num_res_blocks': 0
            },
            'gru': {
                'hidden_size': 128
            },
            'res_lstm': {
                'hidden_size': 128,
                'num_layers': 1
            },
            'tx_like': {
                'hidden_size': 128,
                'num_layers': 1,
                'ffw_multiplier': 4,
                'recurrent_layer': 'lstm',
                'activation': 'gelu'
            }
        },
        'controller_head': {
            'independent': {
                'residual': False
            },
            'autoregressive': {
                'residual_size': 128,
                'component_depth': 0
            },
            'name': 'independent'
        },
        'embed': {
            'player': {
                'xy_scale': 0.05,
                'shield_scale': 0.01,
                'speed_scale': 0.5,
                'with_speeds': False,
                'with_controller': False
            },
            'controller': {
                'axis_spacing': 16,
                'shoulder_spacing': 4
            }
        },
        'policy': {
            'train_value_head': True,
            'delay': 0
        },
        'value_function': {
            'train_separate_network': True,
            'separate_network_config': True,
            'network': {
                'name': 'mlp',
                'mlp': {
                    'depth': 2,
                    'width': 128,
                    'dropout_rate': 0.0
                },
                'lstm': {
                    'hidden_size': 128,
                    'num_res_blocks': 0
                },
                'gru': {
                    'hidden_size': 128
                },
                'res_lstm': {
                    'hidden_size': 128,
                    'num_layers': 1
                },
                'tx_like': {
                    'hidden_size': 128,
                    'num_layers': 1,
                    'ffw_multiplier': 4,
                    'recurrent_layer': 'lstm',
                    'activation': 'gelu'
                }
            }
        },
        'max_names': 16,
        'expt_root': str(models_path),
        'expt_dir': None,  # Will be auto-generated
        'tag': experiment_name,
        'restore_pickle': None,
        'is_test': False,
        'version': 3
    }
    
    return config

def run_training_command(cmd, env, cwd):
    """
    Run a training command and return True if successful
    """
    print(f"Command: {' '.join(str(arg) for arg in cmd)}")
    
    try:
        # Run the command with real-time output
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Capture output
        output_lines = []
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(line.rstrip())
                output_lines.append(line.rstrip())
        
        return_code = process.poll()
        
        if return_code == 0:
            print(f"✅ Training completed successfully!")
            return True
        else:
            print(f"❌ Training failed with return code: {return_code}")
            return False
            
    except Exception as e:
        print(f"❌ Command failed with exception: {e}")
        return False

@app.function(
    image=image,
    volumes={
        "/dataset": dataset_volume,
        "/models": models_volume,
    },
    timeout=86400,  # 24-hour timeout for training
    gpu="A10G",  # Fixed deprecated syntax
    retries=modal.Retries(
        max_retries=2,
        backoff_coefficient=2.0,
        initial_delay=1.0,
    ),
)
def train_slippi_ai():
    """
    Train the Slippi AI model with proper configuration
    """
    # Add project root to the path to ensure all imports work correctly
    sys.path.append(project_root_path_str)
    os.chdir(project_root_path_str)

    try:
        print("--- Starting Training on Modal GPU ---")
        
        # Check GPU availability
        try:
            import tensorflow as tf
            print(f"TensorFlow version: {tf.__version__}")
            gpus = tf.config.list_physical_devices('GPU')
            print(f"GPU Available: {gpus}")
            if gpus:
                print("GPU detected and available for training!")
                # Configure GPU memory growth to avoid OOM
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            else:
                print("WARNING: No GPU detected - training will use CPU")
        except ImportError:
            print("TensorFlow not available for GPU check")
        
        dataset_path = Path("/dataset")
        models_path = Path("/models")
        
        # Ensure models directory exists
        models_path.mkdir(exist_ok=True)
        
        # Check dataset more thoroughly
        print(f"Dataset path: {dataset_path}")
        if dataset_path.exists():
            print(f"Dataset contents: {list(dataset_path.iterdir())}")
            
            # Check for data files recursively
            slp_files = list(dataset_path.rglob("*.slp"))
            pkl_files = list(dataset_path.rglob("*.pkl"))
            print(f"Found {len(slp_files)} .slp files and {len(pkl_files)} .pkl files (recursive search)")
            
            if len(slp_files) == 0 and len(pkl_files) == 0:
                print("❌ No training data found! Please upload data to the dataset volume.")
                print("Expected file types: .slp (raw replays) or .pkl (processed data)")
                return
        else:
            print("❌ Dataset directory does not exist!")
            return
        
        # Create training configuration
        config = create_training_config(dataset_path, models_path, "modal_training_v1")
        
        # Set environment variables
        env = dict(os.environ)
        env.update({
            "SLIPPI_DATA_PATH": str(dataset_path),
            "SLIPPI_MODEL_PATH": str(models_path),
            "PYTHONPATH": f"{project_root_path_str}:{env.get('PYTHONPATH', '')}",
            "CUDA_VISIBLE_DEVICES": "0",
        })
        
        # Method 1: Try using individual config flags with proper argument format
        print("\n=== Method 1: Using individual config flags ===")
        cmd = [
            sys.executable, "scripts/train.py",
            f"--config.dataset.data_dir={dataset_path}",
            f"--config.expt_root={models_path}",
            f"--config.tag=modal_training_v1",
            f"--config.runtime.max_runtime=3600",
            f"--config.data.batch_size=32",
            f"--config.learner.learning_rate=0.0001",
        ]
        
        success = run_training_command(cmd, env, project_root_path_str)
        if success:
            return
        
        # Method 2: Try with config file (most reliable approach)
        print("\n=== Method 2: Using config file ===")
        config_file = Path(models_path) / "training_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        cmd = [
            sys.executable, "scripts/train.py",
            f"--config_file={config_file}"
        ]
        
        success = run_training_command(cmd, env, project_root_path_str)
        if success:
            return
        
        # Method 3: Try minimal required flags
        print("\n=== Method 3: Using minimal required flags ===")
        cmd = [
            sys.executable, "scripts/train.py",
            f"--config.dataset.data_dir={dataset_path}",
            f"--config.expt_root={models_path}",
        ]
        
        success = run_training_command(cmd, env, project_root_path_str)
        if success:
            return
        
        # Method 4: Try importing and running directly with proper argument handling
        print("\n=== Method 4: Direct Python import with argument simulation ===")
        try:
            # Set up sys.argv to simulate command line arguments
            original_argv = sys.argv.copy()
            sys.argv = [
                "train.py",
                f"--config.dataset.data_dir={dataset_path}",
                f"--config.expt_root={models_path}",
                f"--config.tag=modal_training_v1"
            ]
            
            # Try to import and run the training script directly
            sys.path.insert(0, str(Path(project_root_path_str) / "scripts"))
            
            import train
            if hasattr(train, 'main'):
                print("Running train.main() directly...")
                train.main(sys.argv[1:])  # Pass arguments excluding script name
                print("✅ Direct import training completed!")
                return
            else:
                print("No main function found in train module")
                
        except Exception as e:
            print(f"Direct import failed: {e}")
        finally:
            # Restore original sys.argv
            sys.argv = original_argv
        
        # If all methods fail
        print("\n❌ All training methods failed!")
        print("The script may require specific setup or different approach.")
        print("Check the logs above for specific error messages.")
        raise RuntimeError("All training methods failed")

    except Exception as e:
        print(f"Training error: {e}")
        raise

@app.function(
    image=image,
    volumes={
        "/dataset": dataset_volume,
        "/models": models_volume,
    },
    timeout=300,  # 5 minutes for testing
)
def test_environment():
    """
    Test the environment setup and data availability
    """
    import sys
    import os
    from pathlib import Path
    
    print("=== Environment Test ===")
    
    os.chdir(project_root_path_str)
    sys.path.append(project_root_path_str)
    
    # Test imports
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} available")
        print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
    except Exception as e:
        print(f"❌ TensorFlow import failed: {e}")
    
    try:
        import peppi
        print(f"✅ peppi available")
    except Exception as e:
        print(f"❌ peppi import failed: {e}")
    
    # Test project structure
    scripts_dir = Path(project_root_path_str) / "scripts"
    if scripts_dir.exists():
        print(f"✅ Scripts directory exists")
        train_py = scripts_dir / "train.py"
        if train_py.exists():
            print(f"✅ train.py exists")
        else:
            print(f"❌ train.py not found")
    else:
        print(f"❌ Scripts directory not found")
    
    # Test data more thoroughly
    dataset_path = Path("/dataset")
    if dataset_path.exists():
        files = list(dataset_path.iterdir())
        print(f"✅ Dataset directory exists with {len(files)} items")
        
        # Recursive search for training data
        slp_files = list(dataset_path.rglob("*.slp"))
        pkl_files = list(dataset_path.rglob("*.pkl"))
        print(f"Found {len(slp_files)} .slp files and {len(pkl_files)} .pkl files (recursive)")
        
        if slp_files:
            print(f"Sample .slp files: {slp_files[:3]}")
        if pkl_files:
            print(f"Sample .pkl files: {pkl_files[:3]}")
            
        # Check subdirectories
        for item in files:
            if item.is_dir():
                subfiles = list(item.iterdir())
                print(f"  Subdirectory {item.name}: {len(subfiles)} items")
    else:
        print(f"❌ Dataset directory not found")
    
    # Test models directory
    models_path = Path("/models")
    models_path.mkdir(exist_ok=True)
    print(f"✅ Models directory ready")
    
    return True

@app.function(
    image=image,
    volumes={
        "/dataset": dataset_volume,
    },
    timeout=300,
)
def upload_sample_data():
    """
    Upload sample data to the dataset volume for testing
    """
    print("=== Uploading Sample Data ===")
    
    dataset_path = Path("/dataset")
    dataset_path.mkdir(exist_ok=True)
    
    # Create a sample directory structure if it doesn't exist
    sample_dir = dataset_path / "sample_replays"
    sample_dir.mkdir(exist_ok=True)
    
    # Note: This is just a placeholder - you'll need to upload actual .slp files
    print(f"Created sample directory: {sample_dir}")
    print("⚠️ Remember to upload actual .slp replay files to train the model!")
    
    return True

@app.local_entrypoint()
def main():
    """The main entrypoint for the script."""
    print("🚀 Starting Slippi AI training on Modal...")
    
    # Step 1: Test environment
    print("\n🔧 Step 1: Testing environment...")
    try:
        test_environment.remote()
        print("✅ Environment test completed.")
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return
    
    # Step 2: Start training
    print("\n🏋️ Step 2: Starting training...")
    try:
        train_slippi_ai.remote()
        print("✅ Training completed successfully!")
        print("📊 Check the Modal dashboard for detailed logs: https://modal.com/apps")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("\n💡 Next steps:")
        print("1. Check the detailed logs in Modal dashboard")
        print("2. Verify your dataset is properly uploaded to the volume")
        print("3. Upload .slp replay files to the dataset volume")
        print("4. Consider adjusting the configuration parameters")

if __name__ == "__main__":
    # Modal handles the execution via local_entrypoint
    pass