from __future__ import annotations

import argparse
from ast import arg
import contextlib
import time
from pathlib import Path

import cv2
from loguru import logger

from core.detection import Detection
from datasets.frame_iterator import FrameIterator
from detectors.faster_rcnn import FasterRCNNDetector
from evaluation.mot_writer import MOTWriter
from trackers.sort.tracker import SORTTracker
from visualization.renderer import VideoWriter, display_frame, draw_frame

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Tracker demo")
    
    #Required
    parser.add_argument("--video", type=Path, required=True, help="Path to input video file.")
    
    #Optional
    parser.add_argument("--output_video", type=Path, default=None, help="Path to output annotated video.")
    parser.add_argument("--output_mot", type=Path, default=None, help="Path to MOTChallenge output txt file")

    #Detector
    parser.add_argument("--conf_threshold", type=float, default=0.5, help="Detector confidence threshold")
    parser.add_argument("--device", type=str, default="cuda", help="Device for detector inference (cuda/cpu)")

    #Tracker
    parser.add_argument("--max_age", type=int, default=1, help="Max frames a tracker can go unmatched")
    parser.add_argument("--min_hits", type=int, default=3, help="Min hits before a track is confirmed")
    parser.add_argument("--iou_threshold", type=float, default=0.3, help="Min IoU for association match")

    #Display
    parser.add_argument("--live", action="store_true", help="Display frames live")

    #Profile
    parser.add_argument("--profile", action="store_true", help="Print percomponenet timing breakdown")

    return parser.parse_args()

def get_video_metadata(video_path: Path) -> tuple[float, tuple[int, int]]:
    """
    Reads FPS and frame size from video file without decoding frames.

    Args:
        video_path: Path to the video file.
    
    Returns:
        (fps, (width, height))
    
    Raises:
        OSError: If video file cannot be opened
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Could not open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return (fps, (width, height))

def build_components(args: argparse.Namespace) -> tuple[FasterRCNNDetector, SORTTracker, FrameIterator]:
    """
    Instantiate and warmup detector, tracker, and frame iterator.

    Args: 
        args: Parsed command line arguments.

    Returns:
        (detector, tracker, iterator)
    """
    #Build and warmup detector
    detector = FasterRCNNDetector(
        conf_threshold=args.conf_threshold,
        class_ids=[1],
        device=args.device
    )
    detector.warmup()
    #Build tracker
    tracker = SORTTracker(
        max_age=args.max_age,
        min_hits=args.min_hits,
        iou_threshold=args.iou_threshold
    )
    #Build frame iterator
    iterator = FrameIterator(
        source=args.video,
        detector=detector,
        conf_threshold=args.conf_threshold
    )

    return detector, tracker, iterator

def run_loop(
    iterator: FrameIterator,
    tracker: SORTTracker,
    video_writer: VideoWriter | None,
    mot_writer: MOTWriter | None,
    live: bool,
    detector: FasterRCNNDetector | None,
    profile: bool = False
) -> tuple[int, float, set[int], float, float, float]:
    """
    Main tracking loop.

    Args:
        iterator: FrameIterator yielding frames and detections.
        tracker: SORTTracker instance.
        video_writer: Optional VideoWriter context (already entered).
        mot_writer: Optional MOTWriter context (already entered).
        live: Whether to display frames via cv2.imshow.

    Returns:
        (frames_processed, elapsed_seconds, unique_track_ids, detection_time,
        tracking_time, rendering_time)
    """
    frames_processed = 0
    unique_ids: set[int] = set()
    detection_time = 0.0
    tracking_time = 0.0
    rendering_time = 0.0
    start = time.perf_counter()

    for frame in iterator:
        if profile and detector is not None:
            t0 = time.perf_counter()
            detections = detector.detect(frame.image)
            detection_time += time.perf_counter() - t0
        else:
            detections = frame.detections
        #Tracking
        if profile:
            t0 = time.perf_counter()
        tracks = tracker.update(detections)
        if profile:
            tracking_time += time.perf_counter() - t0
        
        unique_ids.update(track[1] for track in tracks)
        
        #Rendering
        if profile:
            t0 = time.perf_counter()
        annotated_frame = draw_frame(frame.image, tracks)
        if profile:
            rendering_time += time.perf_counter() - t0

        if video_writer is not None:
            video_writer.write(annotated_frame)
        
        if mot_writer is not None:
            mot_writer.write(
                frame_index=frame.index,
                tracks=tracks 
            )
        if live:
            if not display_frame(annotated_frame):
                break
        frames_processed += 1

    elapsed = time.perf_counter() - start
    return (
        frames_processed, 
        elapsed, 
        unique_ids, 
        detection_time, 
        tracking_time, 
        rendering_time
    )    

def print_summary(
    frames_processed: int,
    elapsed: float,
    unique_ids: set[int],
    args: argparse.Namespace,
    detection_time: float = 0.0,
    tracking_time: float = 0.0,
    rendering_time: float = 0.0,
) -> None:
    """
    Print end of run summary.

    Args:
        frames_processed: Total frames processed.
        elapsed: Wall-clock seconds elapsed.
        unique_ids: Set of all track IDs seen during the run.
        args: Parsed args (for output paths)
    """

    avg_fps = frames_processed / elapsed if elapsed > 0 else 0.0
    logger.info(f"\nProcessed {frames_processed} frames in {elapsed:.1f}s - avg {avg_fps:.1f} FPS")
    logger.info(f"Tracked objects: {len(unique_ids)} unique IDs")

    if args.profile and frames_processed > 0:
        logger.info(f"\nPer-frame timing breakdown (average over {frames_processed} frames):")
        logger.info(f"Detection:  {detection_time / frames_processed * 1000:.1f} ms")
        logger.info(f"Tracking update:  {tracking_time / frames_processed * 1000:.1f} ms")
        logger.info(f"Rendering:  {rendering_time / frames_processed * 1000:.1f} ms")
        total = (detection_time + tracking_time + rendering_time) / frames_processed * 1000
        logger.info(f"Total pipeline: {total:.1f} ms -> {1000 / total:.1f} FPS")

    if args.output_video:
        logger.info(f"Output video: {args.output_video}")

    if args.output_mot:
        logger.info(f"Output MOT: {args.output_mot}")

def main() -> None:
    args = parse_args()
    #Read video metadata
    fps, frame_size = get_video_metadata(args.video)

    #Build components

    detector, tracker, iterator = build_components(args)

    #Conditionally open output writers using ExitStack
    with contextlib.ExitStack() as stack:
        vw = stack.enter_context(VideoWriter(
            output_path=args.output_video, 
            fps=fps, 
            frame_size=frame_size,
            )) if args.output_video is not None else None
        mw = stack.enter_context(MOTWriter(
            output_path=args.output_mot
        )) if args.output_mot is not None else None
    
        #Run main loop
        frames_processed, elapsed, unique_ids, \
            det_time, track_time, render_time = run_loop(
            iterator=iterator,
            tracker=tracker,
            video_writer=vw,
            mot_writer=mw,
            live=args.live,
            detector=detector,
            profile=args.profile
        )
    print_summary(
        frames_processed=frames_processed,
        elapsed=elapsed,
        unique_ids=unique_ids,
        args=args,
        detection_time=det_time,
        tracking_time=track_time,
        rendering_time=render_time
    )

if __name__ == "__main__":
    main()


