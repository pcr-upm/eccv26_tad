import os
import subprocess
import argparse
from tqdm import tqdm
import tempfile
import shutil
from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)  # ADDED: For parallel processing


def find_video_files(input_dir):
    """Recursively finds all video files in a directory."""
    video_extensions = (".mp4", ".mkv", ".mov", ".avi", ".flv", ".wmv")
    video_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.startswith(".") or file.startswith("tmp"):
                continue
            if file.lower().endswith(video_extensions):
                video_files.append(os.path.join(root, file))
    return video_files


def resize_video(input_path, output_path, height, crf, preset):
    """
    Resizes a video file in-place by writing to a temporary file
    and replacing the original upon success. This prevents data loss
    if the ffmpeg process is interrupted or fails.

    Returns True on success, False on failure.
    """
    # Ensure output directory exists
    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    _, file_ext = os.path.splitext(output_path)

    temp_fd, temp_output_path = tempfile.mkstemp(suffix=file_ext, dir=out_dir)
    os.close(temp_fd)

    command = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vf",
        f"scale=-2:{height}",
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        preset,
        "-c:a",
        "copy",
        "-loglevel",
        "error",
        temp_output_path,
    ]

    try:
        subprocess.run(command, check=True, capture_output=True)
        # Move temporary output into final output path
        os.replace(temp_output_path, output_path)

        # If there is a corresponding annotation .json file next to the input, copy it to output dir
        try:
            in_dir = os.path.dirname(input_path) or "."
            base = os.path.splitext(os.path.basename(input_path))[0]
            json_candidates = []
            # Primary pattern: replace '_rgb_body' with '_rgb_ann_distraction'
            if base.endswith("_rgb_body"):
                ann_base = base[: -len("_rgb_body")] + "_rgb_ann_distraction"
                json_candidates.append(os.path.join(in_dir, ann_base + ".json"))
            # Fallback: same basename + .json
            json_candidates.append(os.path.join(in_dir, base + ".json"))

            for json_src in json_candidates:
                if os.path.exists(json_src):
                    shutil.copy2(json_src, os.path.dirname(output_path) or ".")
                    break
        except Exception:
            # Non-fatal: proceed even if copying fails
            pass
        return True
    except subprocess.CalledProcessError as e:
        # Print error details from ffmpeg if it fails
        print(f"\n  [ERROR] ffmpeg failed for {os.path.basename(input_path)}.")
        try:
            print(f"  ffmpeg stderr: {e.stderr.decode('utf-8').strip()}")
        except Exception:
            pass
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
        return False
    except FileNotFoundError:
        print(
            "\n  [ERROR] ffmpeg command not found. Please ensure ffmpeg is installed and in your system's PATH."
        )
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
        # This error is critical, so we might want to signal the main process to stop.
        # For simplicity here, we let other jobs finish.
        return False
    except Exception as e:
        print(
            f"\n  [ERROR] An unexpected error occurred for {os.path.basename(input_path)}: {e}"
        )
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
        return False


def main():
    """Main function to parse arguments and process videos."""
    # ADDED: Get number of available CPU cores for a sensible default
    default_workers = os.cpu_count()
    print(default_workers)
    parser = argparse.ArgumentParser(
        description="Recursively find and resize all videos in a directory in parallel, OVERWRITING the original files.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="The root directory containing videos.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional directory to write resized videos to. If provided, directory structure from `input-dir` is preserved. If omitted, originals are overwritten.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="The new height for the videos in pixels. Width is auto-calculated to preserve aspect ratio.",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=23,
        help="Quality setting (Constant Rate Factor) for the video codec (0-51, lower is better).",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="fast",
        choices=[
            "ultrafast",
            "superfast",
            "veryfast",
            "faster",
            "fast",
            "medium",
            "slow",
            "slower",
            "veryslow",
        ],
        help="Encoding speed preset. Slower presets provide better compression.",
    )
    # ADDED: Command-line argument to control the number of parallel processes
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help=f"Number of videos to process in parallel. Defaults to the number of CPU cores ({default_workers}).",
    )
    args = parser.parse_args()

    # Safety prompt: vary message depending on whether outputs will overwrite originals
    will_overwrite = False
    if args.output_dir is None:
        will_overwrite = True
    else:
        # If output_dir points to the same location as input_dir, treat as overwrite
        if os.path.abspath(os.path.normpath(args.output_dir)) == os.path.abspath(
            os.path.normpath(args.input_dir)
        ):
            will_overwrite = True

    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!                        W A R N I N G                       !!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    if will_overwrite:
        print("\nThis script will PERMANENTLY overwrite the original video files")
        print(f"in the directory: '{os.path.abspath(args.input_dir)}'")
    else:
        print("\nThis script will NOT modify your original video files.")
        print(
            f"Resized videos will be written to: '{os.path.abspath(args.output_dir)}'"
        )

    print(f"It will run with up to {args.workers} parallel processes.")
    print("\nThis action cannot be undone for overwritten files.")
    print("It is recommended to have a backup of your files before proceeding.")

    if will_overwrite:
        confirm = input(
            '\nTo confirm you understand and wish to proceed, type "OVERWRITE": '
        )
        if confirm != "OVERWRITE":
            print("\nConfirmation not received. Aborting script.")
            exit(0)
        print("\nConfirmation received. Starting process...")
    else:
        confirm = input('\nTo confirm you understand and wish to proceed, type "GO": ')
        if confirm != "GO":
            print("\nConfirmation not received. Aborting script.")
            exit(0)
        print("\nConfirmation received. Starting process...")

    video_files = find_video_files(args.input_dir)
    # Filter removed to support general video files
    # filtered = [
    #     p
    #     for p in video_files
    #     if os.path.splitext(os.path.basename(p))[0].endswith("_rgb_body")
    # ]
    # if not filtered:
    #     print("No video files ending with '_rgb_body' found. Exiting.")
    #     exit(0)
    # video_files = filtered

    print(
        f"Found {len(video_files)} video(s). Starting resizing process with {args.workers} worker(s)..."
    )

    success_count = 0
    fail_count = 0

    # Prepare output paths for each input file
    jobs = []  # list of tuples (input_path, output_path)
    for path in video_files:
        if args.output_dir:
            # preserve directory structure relative to input_dir
            try:
                rel = os.path.relpath(path, args.input_dir)
            except Exception:
                rel = os.path.basename(path)
            out_path = os.path.join(args.output_dir, rel)
        else:
            out_path = path
        jobs.append((path, out_path))

    # --- Process videos in parallel ---
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit all jobs to the executor
        future_to_video = {
            executor.submit(
                resize_video, in_path, out_path, args.height, args.crf, args.preset
            ): in_path
            for in_path, out_path in jobs
        }

        # Create a progress bar that updates as jobs are completed
        progress_bar = tqdm(
            as_completed(future_to_video),
            total=len(video_files),
            desc="Resizing Videos (parallel)",
            unit="file",
        )

        for future in progress_bar:
            result = (
                future.result()
            )  # Get the return value (True/False) from the function
            if result:
                success_count += 1
            else:
                fail_count += 1

    print("\n------------------------------------------------------------")
    print("Processing complete.")
    print(f"Successfully processed: {success_count} video(s)")
    if fail_count > 0:
        print(
            f"Failed to resize: {fail_count} video(s). These files were not modified."
        )
    print(
        "Original video files have been replaced with their resized versions when --output-dir was not provided."
    )


if __name__ == "__main__":
    main()
"""
python tools/resize_vid.py --input-dir /datasets/dmd/dmd/ --output-dir /datasets/dmd/dmd_resized/ --height 256 --crf 23 --preset fast --workers 24
python tools/resize_vid.py --input-dir /media/ricardo/data/datasets/ATTACH/raw_attach_dataset/color --output-dir /media/ricardo/data/datasets/ATTACH/raw_attach_dataset/color --height 480 --crf 23 --preset fast --workers 12
"""
