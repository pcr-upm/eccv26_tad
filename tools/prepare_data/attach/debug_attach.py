import os
import json
import argparse
from tqdm import tqdm

def create_debug_subset(input_json_path, video_folder_path, output_json_path):
    """
    Filters a JSON annotation file to include only entries for videos
    that are present in a local directory.

    Args:
        input_json_path (str): Path to the full annotation JSON file.
        video_folder_path (str): Path to the directory containing video subfolders
                                (e.g., '.../raw_attach_dataset/color/').
        output_json_path (str): Path to save the new, filtered debug JSON file.
    """
    # --- 1. Scan the local video folder for existing video directories ---
    print(f"Scanning for local video folders in: {video_folder_path}")
    try:
        # We get the names of the subdirectories, which correspond to the video names
        local_videos = set(os.listdir(video_folder_path))
        print(f"Found {len(local_videos)} local video folders.")
    except FileNotFoundError:
        print(f"Error: The specified video directory does not exist: {video_folder_path}")
        return

    # --- 2. Load the full annotation file ---
    print(f"Loading full annotation file from: {input_json_path}")
    try:
        with open(input_json_path, 'r') as f:
            full_annotations = json.load(f)
        
        # Safely access the 'database' key
        original_database = full_annotations.get('database', {})
        if not original_database:
            print("Error: The input JSON does not contain a 'database' key or it is empty.")
            return
            
    except FileNotFoundError:
        print(f"Error: The input annotation file does not exist: {input_json_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: The input annotation file is not a valid JSON: {input_json_path}")
        return

    # --- 3. Filter the database and build the new one ---
    debug_database = {}
    print("Filtering annotations to match local videos...")

    for video_name, video_info in tqdm(original_database.items(), desc="Filtering"):
        # The key of each entry in the database is the video name (e.g., '00__0__spike')
        if video_name in local_videos:
            debug_database[video_name] = video_info

    # --- 4. Save the new debug annotation file ---
    total_original = len(original_database)
    total_kept = len(debug_database)
    print(f"\nFiltering complete. Kept {total_kept} out of {total_original} video entries.")
    
    if total_kept == 0:
        print("Warning: No matching videos found. The output file will be empty.")

    # Prepare the final JSON structure
    output_data = {"database": debug_database}

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_json_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving debug annotation file to: {output_json_path}")
    with open(output_json_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Create a smaller, debug-friendly annotation file by filtering "
                    "against a directory of locally available videos."
    )
    parser.add_argument(
        '--input_json',
        type=str,
        required=True,
        help="Path to the full annotation file (e.g., 'attach_person_split_ann.json')."
    )
    parser.add_argument(
        '--video_dir',
        type=str,
        required=True,
        help="Path to the root folder containing the video subdirectories "
             "(e.g., '/path/to/raw_attach_dataset/color/')."
    )
    parser.add_argument(
        '--output_json',
        type=str,
        required=True,
        help="Path to save the new, smaller debug annotation file (e.g., 'attach_debug_ann.json')."
    )

    args = parser.parse_args()

    create_debug_subset(
        input_json_path=args.input_json,
        video_folder_path=args.video_dir,
        output_json_path=args.output_json
    )