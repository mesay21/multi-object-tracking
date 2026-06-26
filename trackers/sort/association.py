from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from core.detection import Detection
from core.geometry import iou_batch

def associate(
    trackers: list[Detection],
    detections: list[Detection],
    iou_threshold: float = 0.3
) -> tuple[list[tuple[int, int]], list[int], list[int]]:

    """
    Associate predicted tracks to detections using IoU and Hungarian assignment.
    
    Args:
        tracks: Predicted track boxes as Detection object, length N.
        detections: Detected boxes from the detector, length M.
        iou_threshold: Minimum IoU to accept a match, Default  0.3.
    
    Returns:
        matched: List of (track_idx, detection_idx) pairs.
        unmatched_detections: Detection indices with no assigned tracks
        unmatched_tracks: Track indices with no assigned detection.
    """
    #No trackers/detections
    if len(trackers) == 0 or len(detections) == 0:
        return (
            [], 
            list(range(len(detections))), 
            list(range(len(trackers)))
        )
    
    #Build boxes array from Detections
    tracker_boxes = np.stack([track.to_xyxy() for track in trackers])
    detection_boxes = np.stack([det.to_xyxy() for det in detections])

    #IoU matrix
    iou_matrix = iou_batch(tracker_boxes, detection_boxes)

    #Solve assignment 
    row_idx, col_idx = linear_sum_assignment(1.0 - iou_matrix)
    matches = []
    matched_tracks = set()
    matched_detections = set()
    for track_idx, detection_idx in zip(row_idx, col_idx):
        if iou_matrix[track_idx][detection_idx] >= iou_threshold:
            matches.append((track_idx, detection_idx))
            matched_detections.add(detection_idx)
            matched_tracks.add(track_idx)
    
    unmatched_detections = [idx for idx in range(len(detections)) if idx not in matched_detections]
    unmatched_tracks = [idx for idx in range(len(trackers)) if idx not in matched_tracks]

    return matches, unmatched_detections, unmatched_tracks
