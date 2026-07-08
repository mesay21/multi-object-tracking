from __future__ import annotations

from pathlib import Path

from core.detection import Detection

class MOTWriter:
    """
    Writes tracker output to a file in MOTChallenge format.
    One row per tracked box per frame:
        frame, id, bb_left, bb_top, bb_width, bb_height, conf, -1, -1, -1
    
    Coordinates are in [x, y, w, h] top-left origin format.

    Args:
        output_path: Path to the output .txt file.
    
    Usage:
        with MOTWriter("outputs/seq1.txt") as writer:
            for frame in iterator:
                tracks = tracker.update(frame.detections)
                writer.write(frame.index, tracks)
    """

    def __init__(self, output_path: str | Path) -> None:
        self.output_path = output_path
        self._file = None
    
    def __enter__(self) -> MOTWriter:
        self._file = open(self.output_path, "w", encoding="utf-8")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._file:
            self._file.close()
    
    def write(
        self,
        frame_index: int,
        tracks: list[tuple[Detection, int]]
    ) -> None:
        """
        Write all confirmed tracks for one frame.

        Args:
            frame_index: 1-based frame index.
            tracks: list of (Detection, track_id) from Tracker.update().
        """
        for detection, track_id in tracks:
            result_str = self._format_row(
                frame_index=frame_index,
                track_id=track_id,
                detection=detection
            )
            self._file.write(result_str)
    
    def _format_row(
        self,
        frame_index: int,
        track_id: int,
        detection: Detection
    ) -> str:
        """
        Format a single row as a MOTChallenge CSV string.

        Returns:
            String: "frame,id,bb_left,bb_top,bb_width,bb_height,conf,-1,-1,-1"
        """
        x, y, w, h = detection.to_xywh()
        result_str = f"{frame_index},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{detection.score},-1,-1,-1\n"
        return result_str

