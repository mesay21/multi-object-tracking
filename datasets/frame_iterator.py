"""
Frame iterator for Video file.
Yields one Frame per iteration containing BGR image and a list of Detection
objects. Detection come from a BaseDetector running on each frame.

Usage:
    iterator = FrameIterator(
                source=<video file path>
                detector=<BaseDetector>
            )
    for frame in iterator:
        print(frame.index, len(frame.detections))
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Literal

import cv2
import numpy as np
from loguru import logger

from core.detection import Detection
from detectors.base import BaseDetector


@dataclass
class Frame:
    """
    A single frame and associated detections.

    Attributes:
        index: frame index (1-based)
        image: BGR uint8 np.ndarrray of shape (H, W, 3)
        detection: Detections for the frame.
    """

    index: int
    image: np.ndarray
    detections: list[Detection] = field(default_factory=list)

class FrameIterator:
    """
    Iterates over frames from a video file.

    Args:
        source: path to a video file.
        detector: A BaseDetector
        conf_threshold: Minimum detection confidence to keep. (Note if detector has its own threshold, this provides additional filter on top.)
    
    Raises:
        ValueError: If required arguments are missing or source path is invlaid.
        FileNotFoundError: If files are missing
    """

    def __init__(
        self,
        source: str | Path,
        detector: BaseDetector,
        conf_threshold: float = 0.0
    ) -> None:
        self.source = Path(source)
        self.detector =  detector
        self.conf_threshold = conf_threshold

        self._validate()
    
    def __iter__(self) -> Generator[Frame, None, None]:
        yield from self._iter_video()
    
    def __len__(self) -> int:
        """
        Return total number of frames (reads frame count from the VideoCapture metadata)
        """
        cap = cv2.VideoCapture(str(self.source))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return count
    
    def _iter_video(self) -> Generator[Frame, None, None]:
        cap = cv2.VideoCapture(str(self.source))

        if not cap.isOpened():
            raise OSError(f"Could not open video file: {self.source}")
        
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        logger.info(
            f"Video iterator: ~{total} frames @ {fps}, source={self.source.name}"
        )
        frame_index = 1

        try:
            while True:
                ret, image = cap.read()
                if not ret:
                    break
                
                detections = self.detector.detect(image)

                if self.conf_threshold > 0:
                    detections = [d for d in detections if d.score >= self.conf_threshold]
                
                yield Frame(index=frame_index, image=image, detections=detections)
                frame_index += 1
        finally:
            cap.release()
            logger.info(f"Video capture released after {frame_index} frames.")
    
    def _validate(self) -> None:

        if not self.source.exists():
            raise FileNotFoundError(f"Source path does not exist: {self.source}")
        
    def __repr__(self) -> str:
        return (
            f"FrameIterator("
            f"source={self.source.name!r}, "
            f"conf_threshold={self.conf_threshold}"
        )        

