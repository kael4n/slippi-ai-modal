# scripts/prepare_replays.py
# FINAL VERSION: A self-contained script to correctly parse replays.
# This version includes the definitive fix for the path object TypeError.

import os
import sys
import pickle
import peppi_py
from pathlib import Path
from tqdm import tqdm
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

def find_slp_files(search_path: Path) -> list[Path]:
    """Recursively finds all .slp files in a given directory."""
    print(f"Searching for .slp files in: {search_path}")
    if not search_path.is_dir():
        print(f"Error: Provided path '{search_path}' is not a directory.")
        return []
    
    slp_files = list(search_path.rglob("*.slp"))
    print(f"Found {len(slp_files)} .slp files.")
    return slp_files

def parse_and_save_replay(slp_path: Path, output_root: Path) -> str:
    """
    Parses a single .slp file using peppi_py and saves the game object as a pickle file.
    """
    try:
        # Define the output path for the parsed file
        game_id = slp_path.stem
        output_dir = output_root / "games" / game_id[:2]
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{game_id}.pkl"

        # FINAL, DEFINITIVE FIX: Convert the Path object to a string for the library.
        game = peppi_py.read_slippi(str(slp_path))

        # Save the parsed game object as a pickle file
        with open(output_path, 'wb') as f:
            pickle.dump(game, f, protocol=pickle.HIGHEST_PROTOCOL)

        return f"Successfully processed {slp_path.name}"
    except Exception as e:
        # Catch any error during parsing (e.g., from a corrupted file)
        return f"ERROR processing {slp_path.name}: {e}"

def main():
    """
    Main function to orchestrate the finding and parsing of .slp files.
    """
    project_root = Path(__file__).resolve().parent.parent
    replay_directory_path = project_root.parent / "SSBM" / "replays" / "2024-10"
    output_directory = project_root / "slp_parsed"

    # --- Safety Check for replay path ---
    if not replay_directory_path.exists():
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! ERROR: The specified replay directory does not exist.              !!!")
        print(f"!!! Path: {replay_directory_path.resolve()}")
        print("!!! Please make sure your replay folder structure is correct.          !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return

    # Clear the output directory before starting
    if output_directory.exists():
        print(f"Clearing old results from: {output_directory}")
        shutil.rmtree(output_directory)
    output_directory.mkdir(exist_ok=True)

    slp_files = find_slp_files(replay_directory_path)

    if not slp_files:
        print("No .slp files found. Exiting.")
        return

    print(f"\nReplays will be processed and saved in: {output_directory}")
    
    # Restoring multi-process execution for speed
    cores = os.cpu_count()
    num_workers = max(1, cores - 2) if cores else 2
    print(f"Using {num_workers} workers for parsing...")

    success_count = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(parse_and_save_replay, slp_file, output_directory): slp_file for slp_file in slp_files}
        
        progress_bar = tqdm(total=len(slp_files), desc="Processing Replays")
        for future in as_completed(futures):
            result = future.result()
            if "ERROR" not in result:
                success_count += 1
            progress_bar.update(1)
        progress_bar.close()

    print("\n--- Preprocessing Complete ---")
    print(f"Successfully processed {success_count} / {len(slp_files)} files.")
    
    if success_count > 0:
        print("Parsed data is located in: C:\\dev\\slippi-ai\\slp_parsed")
        print("You are now ready to upload this data to Modal!")
    else:
        print("WARNING: No files were processed successfully. If errors persist, please show the output.")


if __name__ == "__main__":
    main()