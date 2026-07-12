"""
Wraper for TrackEval interface.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate tracker output on PersonPath22 using TrackEval."
    )
    #Required paths
    parser.add_argument(
        "--tracker_folder",
        type=Path,
        required=True,
        help="Root folder containing tracker MOT txt files."
    )
    parser.add_argument(
        "--gt_folder",
        type=Path,
        required=True,
        help="PersonPath22 ground truth folder"
    )
    parser.add_argument(
        "--trackeval_dir",
        type=Path,
        required=True,
        help="Path to cloned TrackEval repository"
    )
    #Tracker config
    parser.add_argument(
        "--tracker_name",
        type=str,
        default="sort",
        help="Name for the tracker run. Default: sort."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on. Default: test."
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["HOTA", "CLEAR", "Identity"],
        help="TrackEval metric families to compute. Default: HOTA CLEAR Identity"
    )
    #Parallelism
    parser.add_argument(
        "--use_parallel",
        action="store_true",
        help="Run evaluation in parallel"
    )
    parser.add_argument(
        "--num_cores",
        type=int,
        default=1,
        help="Number of parallel cores. Default: 1."
    )

    return parser.parse_args()

def validate_paths(args: argparse.Namespace) -> None:
    """
    Validate required paths exist before invoking TrackEval.

    Args:
        args: Parsed command line arguments.
    
    Raises:
        FileNotFoundError: If any required path doesn't exist.
    """
    if not args.tracker_folder.exists():
        raise FileNotFoundError(f"Tracker folder not found: {args.tracker_folder}")

    if not args.gt_folder.exists():
        raise FileNotFoundError(f"Ground truth folder not found: {args.gt_folder}")

    script: Path = args.trackeval_dir / "scripts" / "run_person_path_22.py"
    if not script.exists():
        raise FileNotFoundError(f"File run_person_path_22.py is missing from: {args.trackeval_dir}")

def build_trackeval_command(args: argparse.Namespace) -> list[str]:
    """
    Build subprocess command to invoke TrackEval.

    Args:
        args: Parsed command line arguments.
    
    Returns:
        List of string tokens forming the command.
    """
    script: Path = args.trackeval_dir / "scripts" / "run_person_path_22.py"

    cmd = [
        sys.executable,
        str(script),
        "--GT_FOLDER", str(args.gt_folder),
        "--TRACKERS_FOLDER", str(args.tracker_folder),
        "--TRACKERS_TO_EVAL", args.tracker_name,
        "--SPLIT_TO_EVAL", args.split,
        "--METRICS", *args.metrics,
        "--USE_PARALLEL", str(args.use_parallel),
        "--NUM_PARALLEL_CORES", str(args.num_cores),
        "--PRINT_CONFIG", "True"
    ]

    return cmd

def run_trackeval(cmd: list[str]) -> int:
    """
    Invoke TrackEval as a subprocess.

    Args:
        cmd: Command tokens to execute.
    
    Returns:
        Exit code from TrackEval process.
    """
    print(f"Running: {" ".join(cmd)}")
    completed_process = subprocess.run(cmd)

    return completed_process.returncode

def main() -> None:
    args = parse_args()
    validate_paths(args=args)
    cmd = build_trackeval_command(args=args)
    status = run_trackeval(cmd)
    sys.exit(status)

if __name__ == "__main__":
    main()



