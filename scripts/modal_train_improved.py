import os
import subprocess
import modal
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# --- Global Definitions ---
DATASET_VOLUME_NAME = "slippi-ai-dataset-doesokay"
MODELS_VOLUME_NAME = "slippi-ai-models-doesokay"
PROJECT_ROOT = "/root/slippi-ai"
REPO_URL = "https://github.com/vladfi1/slippi-ai.git"
PEPPI_PY_COMMIT_URL = "git+https://github.com/hohav/peppi-py.git@8c02a4659c3302321dfbfcf2093c62f634e335f7"

# --- Image Definition with Additional Patches ---
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "build-essential", "pkg-config", "libssl-dev", "curl")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
    )
    .env({
        "PATH": "/root/.cargo/bin:$PATH",
        "TF_ENABLE_ONEDNN_OPTS": "0",
    })
    .run_commands(
        f"git clone {REPO_URL} {PROJECT_ROOT}",
    )
    .workdir(PROJECT_ROOT)
    .run_commands(
        ". ~/.cargo/env && pip install --upgrade pip setuptools wheel",
        ". ~/.cargo/env && pip install maturin python-dotenv ml-collections",
        f". ~/.cargo/env && pip install --no-build-isolation '{PEPPI_PY_COMMIT_URL}'",
        "pip install -r requirements.txt",
        "pip install -e .",
        # Original dataset path fix
        'sed -i "s|train_lib.train(config)|config.dataset.data_dir = \\"/dataset\\"; train_lib.train(config)|" scripts/train.py',
        
        # NEW: Add a patch to handle the meta data format issue
        # This creates a backup and patches the data.py file to handle tuple metadata
        'cp slippi_ai/data.py slippi_ai/data.py.backup',
        '''sed -i 's/@property/# PATCHED: Handle tuple metadata\\n    @property/g' slippi_ai/data.py''',
        '''sed -i '/def main_player(self):/,/return self.meta.p1 if self.swap else self.meta.p0/{
            s/return self.meta.p1 if self.swap else self.meta.p0/# PATCHED: Handle tuple metadata\\n        if isinstance(self.meta, tuple):\\n            return self.meta[1] if self.swap else self.meta[0]\\n        else:\\n            return self.meta.p1 if self.swap else self.meta.p0/
        }' slippi_ai/data.py''',
        
        # Also patch the other_player property
        '''sed -i '/def other_player(self):/,/return self.meta.p0 if self.swap else self.meta.p1/{
            s/return self.meta.p0 if self.swap else self.meta.p1/# PATCHED: Handle tuple metadata\\n        if isinstance(self.meta, tuple):\\n            return self.meta[0] if self.swap else self.meta[1]\\n        else:\\n            return self.meta.p0 if self.swap else self.meta.p1/
        }' slippi_ai/data.py''',
        
        # Add a verification step
        'echo "=== Verifying patches ===" && grep -A 10 "Handle tuple metadata" slippi_ai/data.py || echo "Patch verification failed"'
    )
)

# --- Modal App and Volumes ---
app = modal.App("slippi-ai-trainer-patched")
dataset_volume = modal.Volume.from_name(DATASET_VOLUME_NAME)
models_volume = modal.Volume.from_name(MODELS_VOLUME_NAME, create_if_missing=True)

@app.function(
    image=image,
    volumes={
        "/dataset": dataset_volume,
        "/models": models_volume,
    },
    timeout=86400,
    gpu="A10G",
    secrets=[modal.Secret.from_dict({
        "WANDB_API_KEY": os.getenv("WANDB_API_KEY", ""),
        "WANDB_PROJECT": os.getenv("WANDB_PROJECT", "slippi-ai-training"),
        "WANDB_ENTITY": os.getenv("WANDB_ENTITY", ""),
    })] if os.getenv("WANDB_API_KEY") else []
)
def train():
    """Runs the fully patched slippi-ai training script."""
    os.chdir(PROJECT_ROOT)
    
    print("--- 🚀 Launching Fully Patched Training on Modal ---")
    print(f"WandB Project: {os.getenv('WANDB_PROJECT', 'not set')}")
    print(f"Dataset path: /dataset")
    print(f"Models path: /models")
    
    # Verify patches were applied
    print("=== Verifying Patches ===")
    try:
        with open("slippi_ai/data.py", "r") as f:
            content = f.read()
            if "Handle tuple metadata" in content:
                print("✅ Data format patch applied successfully")
            else:
                print("❌ Data format patch not found")
    except Exception as e:
        print(f"❌ Error checking patches: {e}")
    
    # Check dataset structure
    print("=== Dataset Structure ===")
    dataset_path = Path("/dataset")
    if dataset_path.exists():
        contents = list(dataset_path.iterdir())
        print(f"Dataset contents: {[c.name for c in contents]}")
        
        # Check for required files
        meta_file = dataset_path / "meta.json"
        parsed_dir = dataset_path / "Parsed"
        
        if meta_file.exists():
            print("✅ meta.json found")
        else:
            print("❌ meta.json not found")
            
        if parsed_dir.exists():
            parsed_files = list(parsed_dir.glob("*.json"))
            print(f"✅ Found {len(parsed_files)} parsed files")
        else:
            print("❌ Parsed directory not found")
    else:
        print("❌ Dataset directory not found")
    
    # Ensure models directory exists
    os.makedirs("/models", exist_ok=True)
    os.makedirs("/models/logs", exist_ok=True)
    
    command = ["python", "scripts/train.py"]
    print(f"Executing command: {' '.join(command)}")
    
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{env.get('PYTHONPATH', '')}"
    
    with subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True
    ) as p:
        for line in p.stdout:
            print(line, end='')
    
    if p.returncode != 0:
        print(f"\n❌ Training process failed with exit code {p.returncode}")
        raise RuntimeError(f"Training process failed.")
    else:
        print("\n✅ Training process finished successfully.")

@app.local_entrypoint()
def main():
    print("Submitting fully patched training job to Modal...")
    # Verify environment variables are set
    if not os.getenv("WANDB_API_KEY"):
        print("⚠️  WANDB_API_KEY not set - metrics logging may not work")
    
    train.remote()