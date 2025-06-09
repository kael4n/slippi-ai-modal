# slippi_test_data_corrected.py
# Corrected validation script for .pkl files using working image from slippi_WORKING.py

import modal
from slippi_WORKING import image  # Use the prebuilt image with peppi_py 0.6.0

VOLUME_NAME = "slippi-ai-dataset-doesokay"
PKL_DIR = "/dataset/games/Ga"

app = modal.App("slippi-ai-test-data-corrected")
volume = modal.Volume.from_name(VOLUME_NAME)

@app.function(
    image=image,
    volumes={"/dataset": volume},
    timeout=300,
)
def validate_pickles():
    import pickle
    from pathlib import Path

    print(f"🔍 Scanning {PKL_DIR} for .pkl files...")
    path = Path(PKL_DIR)
    files = list(path.glob("*.pkl"))

    print(f"📦 Found {len(files)} .pkl files")
    if not files:
        print("⚠️ No pickle files found.")
        return

    for file in files[:5]:  # Just check a few files
        print(f"📂 Checking: {file.name}")
        try:
            with open(file, "rb") as f:
                data = pickle.load(f)
                print(f"   ✅ Loaded successfully. Type: {type(data)}")

                # Show structure
                if isinstance(data, dict):
                    print(f"   🔑 Keys: {list(data.keys())[:5]}")
                elif isinstance(data, list):
                    print(f"   📏 Length: {len(data)}")
                else:
                    print("   ℹ️ Custom data structure.")

        except Exception as e:
            print(f"   ❌ Failed to load {file.name}: {e}")

@app.local_entrypoint()
def main():
    validate_pickles.remote()
