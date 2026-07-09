from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from core.detection import Detection

def draw_frame(
    image: np.ndarray,
    tracks: list[tuple(Detection, int)]
) -> np.ndarray:
    """
    Draw bboxes and track IDs on a copy of the frame.

    Args:
        image: BGR uint8 np.ndarray of shape (H, W, 3)
        tracks: List of (Detection, track_id) from Tracker.update()
    Returns:
        Annotated copy of the frame. Original is modified.
    """
    img_copy = image.copy()

    for detection, track_id in tracks:
        color = _track_color(track_id)
        thickness = _box_thickness(image)
        x1, y1 = detection.x1, detection.y1
        x2, y2 = detection.x2, detection.y2
        score = detection.score
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)
        label = f"ID: {track_id} | {score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(
            text=label, 
            fontFace=font, 
            fontScale=font_scale, 
            thickness=font_thickness
            )
        #Draw a filled rectangle
        cv2.rectangle(
            img=img_copy, 
            pt1=(x1, y1 - text_h - 10), 
            pt2=(x1 + text_w, y1), 
            color=(0, 0, 0), 
            thickness=-1
        )

        cv2.putText(
            img=img_copy, 
            text=label, 
            org=(x1, y1 - 5), 
            fontFace=font, 
            fontScale=font_scale, 
            color=(255, 255, 255), 
            thickness=font_thickness, 
            lineType=cv2.LINE_AA
        )
    
    return img_copy


def _track_color(track_id: int) -> tuple[int, int, int]:
    """
    Return a consistent BGR color for a given track ID.

    Args:
        track_id: Unique track identifier.

    Returns:
        BGR tuple with values in [50, 255]
    """
    rng = np.random.default_rng(seed=track_id)
    return tuple(rng.integers(50, 255, size=3).tolist())

def _box_thickness(image: np.ndarray) -> int:
    """
    Compute box thickness scaled to frame resolution.

    Args:
        image: BGR frame of shape (H, w, 3)
    
    Returns:
        Integer pixel thickness.
    """
    H, W, _ = image.shape
    thickness = max(2, max(H, W)//500)

    return thickness

class VideoWriter:
    """
    Context manager wrapping cv2.VideoWriter

    Args:
        output_path: Path to output video file.
        fps: Frames per second
        frame_size: (width, height) in pixels - OpenCV convention.
        fourcc: FourCC codec string. Default 'mp4v'.
    Usage example:  
        with VideoWriter("out.mp4", fps=30, frame_size=(1920, 1080)) as writer:
            for frame in iterator:
                writer.write(annotated_frame)
    """
    def __init__(
        self,
        output_path: str | Path,
        fps: int,
        frame_size: tuple(int, int),
        fourcc: str = 'mp4v'
    ) -> None:
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        self.fourcc = fourcc
    
    def __enter__(self) -> VideoWriter:
        ...
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        ...
    
    def write(self, frame: np.ndarray) -> None:
        """
        Write a single BGR frame to the output video.
        Args:
            frame: BGR uint8 np.ndarray of shape (H, W, 3)
        """
        ...