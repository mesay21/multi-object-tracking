"""
Sort tracker implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.detection import Detection
from trackers.base import BaseTracker
from trackers.sort.association import associate
from trackers.sort.kalman import KalmanBoxTracker

@dataclass
class TrackState:
    """
    External lifecycle state for a single track.

    Attributes:
        tracker: KalmanBoxTracker - Kalman filter estimation engine.
        track_id: unique ID assigned to a SORTTrack at birth.
        hits: Total number of successful updates (matched detections) since birth.
        time_since_last_update: Total number of frames since last matched detection match.
    """
    tracker: KalmanBoxTracker
    track_id: int
    hits: int
    time_since_last_update: int = 0

class SORTTracker(BaseTracker):
    """
    SORT multi-object tracker.

    Args:
        max_age: Maximum frames a track can go unmatched before deletion.
        min_hits: Minimum hits before a track is considered confirmed.
        iou_threshold: Minimum IoU to accept association match.
        dt: Time step passed to Kalman filter.
    """

    def __init__(
        self, 
        max_age: int = 1, 
        min_hits: int = 3, 
        iou_threshold: float = 0.3, 
        dt: float = 1.0) -> None:

        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.dt = dt

        self.tracks: list[TrackState] = []
        self.frame_count = 0
        self._next_id = 0

    def update(self, detections: list[Detection]) -> list[tuple[Detection, int]]:
        """
        Process one frame of detections and return confirmed tracks.

        Args:
            detections: List of detections from detector.

        Returns:
            List of (box, track_id) pair of confirmed tracks only.
        """
        #Run prediction step for all tracks
        predicted_tracks = [track.tracker.predict() for track in self.tracks]

        #Associate predictions to detections
        matched, unmatched_detections, unmatched_predictions = associate(
            predicted_tracks, detections, self.iou_threshold
        )

        #Update matched tracks
        for track_idx, detection_idx in matched:
            self.tracks[track_idx].tracker.update(detection=detections[detection_idx])
            self.tracks[track_idx].hits += 1
            self.tracks[track_idx].time_since_last_update = 0  # reset on match
        
        #Increament time since last update for unmatched tracks
        for track_idx in unmatched_predictions:
            self.tracks[track_idx].time_since_last_update += 1
        
        #Birth new tracks for unmatched detections
        for detection_idx in unmatched_detections:
            state = TrackState(
                tracker=KalmanBoxTracker(
                    detection=detections[detection_idx],
                    dt=self.dt
                ),
                hits=1,
                track_id=self._next_id
            )
            self._next_id += 1
            self.tracks.append(state)
        
        #Delete all tracks exceeding max_age
        self.tracks = [track for track in self.tracks if track.time_since_last_update <= self.max_age]

        self.frame_count += 1

        confirmed_tracks = []
        for track in self.tracks:
            if track.hits >= self.min_hits and track.time_since_last_update < 1:
                confirmed_tracks.append((track.tracker.get_state(), track.track_id)) 

        return confirmed_tracks


            