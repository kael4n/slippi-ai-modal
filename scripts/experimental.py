# train_on_modal_production.py
# This script combines a clean structure with the definitive, evidence-based
# patch required to make the slippi-ai training script run correctly.

import sys
import os
import subprocess
import json
from pathlib import Path

import modal

# --- Global Definitions ---
DATASET_VOLUME_NAME = "slippi-ai-dataset-doesokay"
MODELS_VOLUME_NAME = "slippi-ai-models-doesokay"
PROJECT_ROOT = "/root/slippi-ai"
REPO_URL = "https://github.com/vladfi1/slippi-ai.git"

# This specific commit of peppi-py is critical to avoid data format errors.
PEPPI_PY_COMMIT_URL = "git+https://github.com/hohav/peppi-py.git@8c02a4659c3302321dfbfcf2093c62f634e335f7"

# --- Image Definition ---
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
        # Use . instead of source for shell compatibility
        ". ~/.cargo/env && pip install --upgrade pip setuptools wheel",
        ". ~/.cargo/env && pip install maturin",
        # Install the correct, pinned version of peppi-py
        f". ~/.cargo/env && pip install --no-build-isolation '{PEPPI_PY_COMMIT_URL}'",
        "pip install -r requirements.txt",
        "pip install -e .",
        # --- DEFINITIVE FIX ---
        # This sed command surgically modifies the train.py script. It finds the line
        # `train_lib.train(config)` and replaces it with a version that first
        # hardcodes the dataset path using the correct dot-notation syntax.
        'sed -i "s|train_lib.train(config)|config.dataset.data_dir = \\"/dataset\\"; train_lib.train(config)|" scripts/train.py',
    )
)

# --- Modal App and Volumes ---
app = modal.App("slippi-ai-trainer-prod")
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
)
def train():
    """
    Runs the patched slippi-ai training script.
    """
    os.chdir(PROJECT_ROOT)
    sys.path.append(PROJECT_ROOT)

    print("--- 🚀 Launching Patched Training on Modal ---")

    # The command is now simple, as the configuration is patched directly into the file.
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
    print("Submitting definitive training job to Modal...")
    train.remote()