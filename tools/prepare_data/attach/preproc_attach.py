import pandas as pd
import json
import os
import argparse
from tqdm import tqdm


def find_filenames_txt(directory):
    """
    Scans a directory to find the file that ends with '_filenames.txt'.
    """
    if not os.path.isdir(directory):
        return None

    for filename in os.listdir(directory):
        if filename.endswith("_filenames.txt"):
            return os.path.join(directory, filename)

    fallback_path = os.path.join(directory, "filenames.txt")
    if os.path.exists(fallback_path):
        return fallback_path

    return None


def create_annotation_file(
    csv_path, raw_dataset_path, output_path, split_type="person", min_duration=0
):
    """
    Generates a JSON annotation file for the ATTACH dataset in the format
    required by the custom dataset loaders.
    """
    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
        return

    split_column = f"{split_type}_split"
    if split_column not in df.columns:
        print(f"Error: Split column '{split_column}' not found in the CSV.")
        return

    print(f"Using '{split_column}' for train/validation/test subsets.")

    grouped = df.groupby("tape")
    database = {}

    print(f"Processing {len(grouped)} unique videos...")
    for video_name, group in tqdm(grouped, desc="Processing Videos"):

        split_value = group.iloc[0][split_column]
        if "TRAIN" in split_value:
            subset = "train"
        elif "VALIDATION" in split_value:
            subset = "val"
        elif "TEST" in split_value:
            subset = "test"
        else:
            continue

        metadata_path = os.path.join(
            raw_dataset_path, "labels", video_name, "metadata.json"
        )
        try:
            # It's good practice to specify encoding for JSON as well
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            total_duration_sec = metadata["lastmessage"] / 1_000_000_000.0
        except FileNotFoundError:
            print(f"Warning: metadata.json not found for {video_name}. Skipping video.")
            continue
        except json.JSONDecodeError:
            print(
                f"Warning: metadata.json for {video_name} is corrupted. Skipping video."
            )
            continue

        video_color_dir = os.path.join(raw_dataset_path, "color", video_name)
        filenames_path = find_filenames_txt(video_color_dir)

        if filenames_path is None:
            print(
                f"Warning: Could not find a '*_filenames.txt' file in directory '{video_color_dir}'. Skipping video."
            )
            continue

        try:
            # ==================== MODIFIED LINE START ====================
            # Added encoding='latin-1' to handle non-UTF-8 text files robustly.
            with open(filenames_path, "r", encoding="latin-1") as f:
                # ===================== MODIFIED LINE END =====================
                num_frames = sum(1 for line in f)
        except FileNotFoundError:
            print(f"Warning: File at {filenames_path} not found. Skipping video.")
            continue

        annotations = []
        discarded_zero_length_count = 0
        discarded_short_count = 0
        for _, row in group.iterrows():
            start_time_sec = row["start_relative"] / 1_000_000_000.0
            end_time_sec = row["end_relative"] / 1_000_000_000.0
            action_label = row["name"]
            frame_duration = row['frame_duration']

            # --- THE FIX IS HERE ---
            # Filter 1: Zero-length segments (based on time)
            if start_time_sec == end_time_sec:
                discarded_zero_length_count += 1
                continue
            
            # Filter 2: Short segments (based on frame_duration column)
            if frame_duration < min_duration:
                discarded_short_count += 1
                continue
            annotations.append(
                {"segment": [start_time_sec, end_time_sec], "label": action_label}
            )

        database[video_name] = {
            "duration": total_duration_sec,
            "subset": subset,
            "frame": num_frames,
            "annotations": annotations,
        }

    final_json = {"database": database}
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print(f"Saving annotation file to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(final_json, f, indent=2)

    print(f"-> Done! Discarded {discarded_zero_length_count} zero-length and {discarded_short_count} short segments.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a JSON annotation file for the ATTACH dataset."
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to the 'labels_with_splits_AllClasses.csv' file.",
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        required=True,
        help="Path to the root of the 'raw_attach_dataset' directory.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./attach_ann.json",
        help="Path to save the output 'attach_ann.json' file.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="person",
        choices=["person", "camera"],
        help="Which data split to use for defining train/val/test subsets ('person' or 'camera').",
    )
    parser.add_argument(
        "--min_duration",
        type=int,
        default=0,
        help="Minimum frame duration for an action to be included. Actions shorter than this will be discarded.",
    )
    args = parser.parse_args()

    create_annotation_file(
        csv_path=args.csv,
        raw_dataset_path=args.raw_dir,
        output_path=args.output,
        split_type=args.split,
        min_duration=args.min_duration,
    )
"""
python tools/prepare_data/attach/preproc_attach.py     --csv /datasets/ATTACH/141.24.24.111:50021/icra_processed_attach_dataset/labels_with_splits_AllClasses.csv     --raw_dir /datasets/ATTACH/141.24.24.111:50021/raw_attach_dataset/     --output ./attach_person_split_ann.json     --split person --min_duration 16
"""