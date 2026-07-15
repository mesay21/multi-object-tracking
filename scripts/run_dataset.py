"""
Runs tracker on PersonPath22 dataset.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from detectors.faster_rcnn import FasterRCNNDetector
from datasets.frame_iterator import FrameIterator
from evaluation.mot_writer import MOTWriter
from trackers.sort.tracker import SORTTracker


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """

    parser = argparse.ArgumentParser(
        description="RUN Tracker on all videos and write MOTChallenge output files"
    )

    #Required
    parser.add_argument("--videos_dir", type=Path, required=True, help="Directory containing .mp4 files")
    parser.add_argument("--output_dir", type=Path, required=True, help="Root output directory where MOT txt files written")

    #Detector
    parser.add_argument("--conf_threshold", type=float, default=0.5, help="Detector confidence threshold")
    parser.add_argument("--device", type=str, default="cuda", help="device for detector inference cuda/cpu")

    #Tracker
    parser.add_argument("--max_age", type=int, default=1, help="Max frames a tracker can go unmatched")
    parser.add_argument("--min_hits", type=int, default=3, help="Min number of frames before a track is confirmed")
    parser.add_argument("--iou_threshold", type=float, default=0.3, help="Min IoU for association match")

    return parser.parse_args()

def validate_paths(args: argparse.Namespace) -> list[Path]:
    """
    Validate input directory and return sorted list of .mp4 files.

    Args:
        args: Parsed command line arguments

    Returns:
        Sorted list of .mp4 file paths.
    
    Raises:
        FileNotFoundError: If videos_dir doesn't exist
        ValueError: If no .mp4 files found in videos_dir
    """
    if not args.videos_dir.exists():
        raise FileNotFoundError(f"Video directory doesn't exist: {args.videos_dir}")
    video_files = sorted(args.videos_dir.glob("*.mp4"))

    if len(video_files) == 0:
        raise ValueError(f"No video files found in the directory: {args.videos_dir}")
    
    return video_files

def build_detector(args: argparse.Namespace) -> FasterRCNNDetector:
    """
    Build and warmup detector.

    Args:
        args: Parsed command line arguments
    
    Returns:
        Warmed up Detector.
    """
    detector = FasterRCNNDetector(
        conf_threshold=args.conf_threshold,
        class_ids=[1], #Person class ID
        device=args.device
    )
    detector.warmup()

    return detector

def process_video(
    video_path: Path,
    detector: FasterRCNNDetector,
    args: argparse.Namespace,
    output_path: Path,
) -> tuple[int, float]:
    """
    Run Tracker on a single video and write MOT txt output.

    Args:
        video_path: Path to the input .mp4 file.
        detector: Warmed-up detector instance.
        args: Parsed command line arguments
        output_path: Path to write MOT txt files.
    
    Returns:
        (frames_processed, elapsed_seconds)
    """
    tracker = SORTTracker(
        max_age=args.max_age,
        min_hits=args.min_hits,
        iou_threshold=args.iou_threshold
    )
    iterator = FrameIterator(
        source=video_path,
        detector=detector,
        conf_threshold=args.conf_threshold
    )
    start = time.perf_counter()
    frame_count = 0
    with MOTWriter(output_path) as mw:
        for frame in iterator:
            tracks = tracker.update(frame.detections)
            mw.write(
                frame_index=frame.index,
                tracks=tracks
            )
            frame_count += 1
    elapsed = time.perf_counter() - start

    return frame_count, elapsed

def main() -> None:
    args = parse_args()

    #Collect videos
    video_paths = validate_paths(args)
    mot_dir = args.output_dir / "data"
    mot_dir.mkdir(parents=True, exist_ok=True)

    detector = build_detector(args)

    #Process each video
    for video_path in tqdm(video_paths):
        output_file = args.output_dir / "data" / f"{video_path.stem}.txt"
        frames, elapsed = process_video(
            video_path=video_path,
            detector=detector,
            args=args,
            output_path=output_file
        )
        logger.info(f"{frames} frames in {elapsed:.1f} - {frames/elapsed:.1f} FPS")
    logger.info(f"Done. MOT files written to {args.output_dir / 'data'}")    

if __name__ == "__main__":
    main()

