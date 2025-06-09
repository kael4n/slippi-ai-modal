# scripts/train_on_modal.py
# FINAL, ROBUST VERSION: This script builds the environment in the correct order,
# installing all build dependencies before the packages that need them.

import sys
from pathlib import Path

import modal

# --- Global Definitions ---
dataset_volume_name = "slippi-ai-dataset-doesokay"
models_volume_name = "slippi-ai-models-doesokay"
project_root_path_str = "/root/slippi-ai"
repo_url = "https://github.com/vladfi1/slippi-ai.git"
# The specific commit for peppi-py required by slippi-ai
peppi_py_commit_url = "git+https://github.com/hohav/peppi-py.git@8c02a4659c3302321dfbfcf2093c62f634e335f7"


# --- Image Definition (Single, Chained Command) ---
# This is the definitive fix to ensure all build steps run in the correct order and context.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .run_commands(
        "apt-get update && apt-get install -y git",
        f"git clone {repo_url} {project_root_path_str}",
        # DEFINITIVE FIX: Chain all pip installs in the correct order
        # to ensure build dependencies are available when needed.
        (
            f"cd {project_root_path_str} && "
            # 1. Install the Rust build tools first.
            "pip install puccinialin maturin && "
            # 2. Now, install the special peppi-py commit that needs the build tools.
            f"pip install --no-build-isolation '{peppi_py_commit_url}' && "
            # 3. Finally, install the rest of the project's requirements.
            "pip install -r requirements.txt"
        ),
    )
)

# --- Modal App and Volumes ---
app = modal.App("slippi-ai-trainer")
dataset_volume = modal.Volume.from_name(dataset_volume_name)
models_volume = modal.Volume.from_name(models_volume_name, create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",
    volumes={
        "/dataset": dataset_volume,
        "/models": models_volume,
    },
    timeout=86400,  # 24-hour timeout for training
)
def train():
    """
    This function runs on a Modal GPU and executes the training script's logic directly.
    """
    # Add project root to the path to ensure all imports work correctly
    sys.path.append(project_root_path_str)

    from slippi_db.generate_metadata import main as generate_metadata_main
    from scripts.train import main as train_main

    print("--- Starting Training on Modal GPU ---")
    
    dataset_path = Path("/dataset")
    models_path = Path("/models")
    
    print("Generating metadata file...")
    # Simulate command-line arguments for the metadata script
    metadata_argv = [
        "generate_metadata.py", # Script name placeholder
        str(dataset_path),
        f"--output={dataset_path / 'meta.json'}"
    ]
    generate_metadata_main(metadata_argv)
    print("metadata.json generated successfully.")

    # --- Training Configuration ---
    train_argv = [
        "train.py", # Script name placeholder
        f"--name=doesokay_v1",
        f"--data_path={dataset_path}",
        f"--test_ratio=0.1",
        f"--batch_size=64",
        f"--epochs=10",
        f"--save_path={models_path}"
    ]

    print("\n--- Training Configuration ---")
    # Print the arguments for clarity, skipping the script name
    for arg in train_argv[1:]:
        print(arg)
    
    print("\n--- Starting Imitation Learning ---")
    print("This will take a long time. You can close your terminal and monitor progress on the Modal dashboard.")
    
    # Call the main training function directly with the arguments
    train_main(train_argv)

    print("\n--- Training Complete! ---")
    print(f"Trained model artifacts are saved in the '{models_volume_name}' Modal Volume.")

@app.local_entrypoint()
def main():
    """The main entrypoint for the script."""
    print("Submitting training job to Modal...")
    with modal.enable_output():
        with app.run(detach=True):
            train.remote()
    print("Job submitted successfully. You can now close this terminal.")
    print("Monitor progress at https://modal.com/apps")

if __name__ == "__main__":
    main()
