#!/usr/bin/env python3
"""
Preprocess keypoint pickle files to NumPy format for faster data loading.

This script converts raw pose estimation pickle files to preprocessed .npy format,
applying confidence filtering and padding in advance. The resulting files can be
loaded with memory-mapping for efficient cross-worker caching.

Usage:
    python tools/prepare_data/activitynet/preprocess_keypoints.py \
        --input data/activitynet-1.3/pose_results \
        --output data/activitynet-1.3/pose_results_processed

The input directory should contain {video_name}.pkl files from pose estimation.
The output directory will contain:
    - {video_name}.npy: Preprocessed keypoints with shape (T, MAX_PEOPLE, K, 2)
    - metadata.json: Per-video metadata (K, MAX_PEOPLE, T)
"""

import argparse
import json
import pickle
from pathlib import Path
from tqdm import tqdm

import numpy as np


def process_keypoint_pickle(pkl_path: str, confidence_threshold: float = 0.5) -> tuple:
    """
    Process a single keypoint pickle file.
    
    Args:
        pkl_path: Path to the pickle file
        confidence_threshold: Threshold below which keypoints are zeroed out
        
    Returns:
        tuple: (keypoints array, metadata dict) or (None, None) if failed
    """
    try:
        with open(pkl_path, "rb") as f:
            video_results = pickle.load(f)
    except Exception as e:
        print(f"Error loading {pkl_path}: {e}")
        return None, None
    
    if len(video_results) == 0:
        return None, None
    
    # Determine K (number of keypoints per person) from first valid detection
    K = 26  # default
    for frame_res in video_results:
        kps = frame_res.get("keypoints", [])
        if len(kps) > 0 and len(kps[0]) > 0:
            K = kps[0].shape[0]
            break
    
    # Determine MAX_PEOPLE across all frames
    MAX_PEOPLE = 0
    for frame_res in video_results:
        kps = frame_res.get("keypoints", [])
        MAX_PEOPLE = max(MAX_PEOPLE, len(kps))
    
    # Default to 1 person if no detections found
    if MAX_PEOPLE == 0:
        MAX_PEOPLE = 1
    
    # Process each frame
    final_kps = []
    for frame_res in video_results:
        kps = frame_res.get("keypoints", [])
        scores = frame_res.get("scores", [])
        
        frame_person_kps = []
        num_people = len(kps)
        
        for i in range(MAX_PEOPLE):
            if i < num_people:
                kp = kps[i].copy()  # (K, 2) - copy to avoid modifying original
                
                # Apply confidence filtering if scores available
                if len(scores) > i:
                    score = scores[i]  # (K,)
                    low_conf_mask = score < confidence_threshold
                    kp[low_conf_mask] = 0
            else:
                # Padding with zeros for missing people
                kp = np.zeros((K, 2), dtype=np.float32)
            
            frame_person_kps.append(kp)
        
        # Stack people for this frame -> (MAX_PEOPLE, K, 2)
        frame_kps_stacked = np.stack(frame_person_kps, axis=0)
        final_kps.append(frame_kps_stacked)
    
    keypoints = np.stack(final_kps).astype(np.float32)  # (T, MAX_PEOPLE, K, 2)
    
    metadata = {
        "T": keypoints.shape[0],
        "MAX_PEOPLE": MAX_PEOPLE,
        "K": K,
        "confidence_threshold": confidence_threshold,
    }
    
    return keypoints, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess keypoint pickle files to NumPy format"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory containing {video_name}.pkl files",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for preprocessed .npy files",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for keypoint filtering (default: 0.5)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing preprocessed files",
    )
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return 1
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all pickle files
    pkl_files = sorted(input_dir.glob("*.pkl"))
    if len(pkl_files) == 0:
        print(f"No .pkl files found in {input_dir}")
        return 1
    
    print(f"Found {len(pkl_files)} pickle files to process")
    print(f"Output directory: {output_dir}")
    print(f"Confidence threshold: {args.confidence_threshold}")
    
    # Process all files
    all_metadata = {}
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    for pkl_path in tqdm(pkl_files, desc="Processing keypoints"):
        video_name = pkl_path.stem
        npy_path = output_dir / f"{video_name}.npy"
        
        # Skip if already exists and not overwriting
        if npy_path.exists() and not args.overwrite:
            # Load existing metadata if available
            skip_count += 1
            continue
        
        keypoints, metadata = process_keypoint_pickle(
            str(pkl_path),
            confidence_threshold=args.confidence_threshold
        )
        
        if keypoints is None:
            fail_count += 1
            continue
        
        # Save preprocessed keypoints
        np.save(npy_path, keypoints)
        all_metadata[video_name] = metadata
        success_count += 1
    
    # Load existing metadata for skipped files
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            existing_metadata = json.load(f)
        # Merge with new metadata (new takes precedence)
        existing_metadata.update(all_metadata)
        all_metadata = existing_metadata
    
    # Save metadata
    with open(metadata_path, "w") as f:
        json.dump(all_metadata, f, indent=2)
    
    print("\nDone!")
    print(f"  Processed: {success_count}")
    print(f"  Skipped (already exist): {skip_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Metadata saved to: {metadata_path}")
    
    # Print size comparison for one file
    if success_count > 0:
        sample_pkl = pkl_files[0]
        sample_npy = output_dir / f"{sample_pkl.stem}.npy"
        if sample_npy.exists():
            pkl_size = sample_pkl.stat().st_size / 1024 / 1024
            npy_size = sample_npy.stat().st_size / 1024 / 1024
            print("\nSize comparison (sample):")
            print(f"  Pickle: {pkl_size:.2f} MB")
            print(f"  NumPy:  {npy_size:.2f} MB")
    
    return 0


if __name__ == "__main__":
    exit(main())
