# slippi_docker.py
# This script is now simple and clean. All the complex environment building
# is handled by the Dockerfile, which is the robust and correct approach.

import modal

# Create an image by pointing to our Dockerfile.
# Modal will build and cache this for you.
image = modal.Image.from_dockerfile("Dockerfile")

app = modal.App("slippi-ai-docker", image=image)

@app.function(
    volumes={"/data": modal.Volume.from_name("slippi-ai-dataset-doesokay")},
    timeout=3600,
    gpu="any" # Requesting a GPU for ML training
)
def train():
    """
    Validates the environment and dataset, then starts the training process.
    """
    import os
    import sys
    import numpy as np

    PROJECT_ROOT = "/root/slippi-ai"

    print("--- 🚀 System Validation and Training ---")

    # Add project to path to ensure imports work
    sys.path.insert(0, PROJECT_ROOT)

    # Step 1: Final validation of the environment
    print(f"✅ NumPy version: {np.__version__}")
    if np.__version__ != "1.24.3":
        print(f"❌ FATAL: NumPy version is incorrect! Expected 1.24.3, got {np.__version__}.")
        return

    # Dynamically import other libraries to confirm they work
    try:
        import tensorflow as tf
        import jax
        import slippi_ai
        print(f"✅ TensorFlow version: {tf.__version__}")
        print(f"✅ JAX version: {jax.__version__}")
        print(f"✅ slippi_ai module found at: {slippi_ai.__path__}")
        print(f"✅ TensorFlow is using GPU: {tf.config.list_physical_devices('GPU')}")
    except ImportError as e:
        print(f"❌ FATAL: A core library failed to import: {e}")
        return

    # Step 2: Validate dataset access
    print("\n" + "=" * 70)
    print("🗂️  Validating dataset access...")
    data_path = "/data/games/Ga"

    if not os.path.exists(data_path):
        print(f"❌ FATAL: Dataset directory not found at '{data_path}'!")
        return

    pkl_files = [f for f in os.listdir(data_path) if f.endswith('.pkl')]
    print(f"✅ Found {len(pkl_files)} .pkl game files in '{data_path}'.")

    if not pkl_files:
        print(f"⚠️  No game files (.pkl) found in '{data_path}'.")
        return

    print("✅ Environment and dataset are confirmed ready.")

    # Step 3: Start the actual training
    print("\n" + "=" * 70)
    print("🏆 Starting Training Process...")
    print("Placeholder for training script execution.")

@app.local_entrypoint()
def main():
    train.remote()