# scripts/upload_to_modal.py
# FINAL VERSION: This script uploads files one-by-one and includes the definitive
# fix for the Path object vs. string issue that caused silent failures.

import sys
from pathlib import Path
import modal

def main():
    """The main entrypoint for the script."""
    volume_name = "slippi-ai-dataset-doesokay"
    
    project_root = Path(__file__).resolve().parent.parent
    local_data_dir = project_root / "slp_parsed"

    # --- Verification Step ---
    if not local_data_dir.exists() or not any(local_data_dir.rglob('*.pkl')):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"!!! ERROR: The source directory '{local_data_dir}' is  !!!")
        print(f"!!! empty or does not contain any .pkl files.        !!!")
        print("!!! Please run `prepare_replays.py` successfully first.!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        sys.exit(1)

    # Get a handle to the cloud Volume, creating it if it doesn't exist.
    volume = modal.Volume.from_name(volume_name, create_if_missing=True)
    
    print(f"Preparing to upload files from '{local_data_dir}' to Modal Volume '{volume_name}'...")
    
    # Get a list of all files to upload
    local_files = [p for p in local_data_dir.rglob("*.pkl")]
    total_files = len(local_files)
    print(f"Found {total_files} files to upload.")

    try:
        # Use a context manager to handle the batch upload efficiently.
        with volume.batch_upload() as batch:
            for i, local_path in enumerate(local_files):
                # Determine the destination path in the cloud volume
                remote_path = local_path.relative_to(local_data_dir).as_posix()
                
                # FINAL FIX: Convert the local_path (a Path object) to a string before passing it.
                batch.put_file(str(local_path), remote_path)
                
                # Print progress periodically
                if (i + 1) % 100 == 0 or (i + 1) == total_files:
                    print(f"Uploaded {i + 1} / {total_files} files...")

        print("\n--- Upload Complete! ---")
        print(f"Successfully uploaded {total_files} files to the cloud Volume: '{volume_name}'")
        print("You can verify the contents by running: modal volume ls slippi-ai-dataset-doesokay")

    except Exception as e:
        print("\n--- AN ERROR OCCURRED DURING UPLOAD ---")
        print("Error Details:", e)
        print("Please check your connection and ensure you are logged into Modal.")

if __name__ == "__main__":
    main()
