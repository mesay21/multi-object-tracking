"""
Unit tests for trackers.sort.tracker.SORTtracker.

Tests covered:
    - Default state construction
    - New tracks created for unmatched detections
    - Returned tracks are after min_hits frames
    - Unmatched tracks deleted after max_age missed frames
    - Unique monotonic ID's assigned
    - Matched tracks updated and reset correctly
    - Return type is list[tuple[Detection, int]]
    - Frame count increaments properly
    - update() with no detections handeled correctly
    - Multiple objects tracked simultaneously
    - time since last update resets on match
"""

from __future__ import annotations

import pytest

from core.detection import Detection
from trackers.sort.tracker import SORTTracker, TrackState

def make_detection(
    x1: float = 0.0,
    y1: float = 0.0,
    x2: float = 50.0,
    y2: float = 100.0,
    score: float = 0.9
) -> Detection:
    return Detection(x1=x1, y1=y1, x2=x2, y2=y2, score=score, class_id=1)

def far_detection(offset: float =  1000.0) -> Detection:
    """
    Detection guaranteed not to overlap.
    """
    return make_detection(x1=offset, y1=offset, x2=offset + 50.0, y2 = offset + 100.0)

def feed_n_frames(
    tracker: SORTTracker,
    detection: Detection,
    n: int
) -> list:
    """
    Feed the same detection for n consecutive frames.
    """
    results = []

    for _ in range(n):
        results.append(tracker.update([detection]))
    
    return results

class TestInitialization:

    def test_default_params(self):
        t = SORTTracker()
        assert t.max_age == 1
        assert t.min_hits == 3
        assert t.iou_threshold == 0.3
        assert t.dt == 1.0

    def test_custom_initalization(self):
        t = SORTTracker(max_age=5, min_hits=10, iou_threshold=0.5, dt=2.0)
        assert t.max_age == 5
        assert t.min_hits == 10
        assert t.iou_threshold == 0.5
        assert t.dt == 2.0

    def test_empty_trackers_at_start(self):
        assert SORTTracker().tracks == []
    
    def test_frame_count_zero_at_start(self):
        assert SORTTracker().frame_count == 0

    def test_next_id_zero_at_start(self):
        assert SORTTracker()._next_id == 0

class TestTrackBirth:

    def test_new_detections_creates_track(self):
        t = SORTTracker()
        t.update([make_detection(), far_detection()])
        assert len(t.tracks) == 2
    
    def test_new_track_hit_is_one(self):
        t = SORTTracker()
        t.update([make_detection()])
        assert t.tracks[0].hits == 1
    
    def test_new_track_time_since_last_update(self):
        """
        For new track time since last update should be zero
        """
        t = SORTTracker()
        t.update([make_detection()])
        assert t.tracks[0].time_since_last_update == 0
    
    def test_empty_detection_creates_no_tracks(self):
        t = SORTTracker()
        t.update([])
        assert len(t.tracks) == 0

class TestTrackConfirmation:

    def test_track_not_comfirmed_before_min_hits(self):
        t = SORTTracker(min_hits=3, max_age=5)
        det = make_detection()
        for _ in range(2):
            result = t.update([det])
            assert result == []
    
    def test_track_confirmed_at_min_hits(self):
        t = SORTTracker(min_hits=3, max_age=5)
        det = make_detection()
        results = feed_n_frames(t, det, n=3)
        assert len(results[-1]) == 1
    
    def test_track_confirmed_after_min_hits(self):
        t = SORTTracker(min_hits=3, max_age=5)
        det = make_detection()
        results = feed_n_frames(t, det, n=5)
        assert len(results[-1]) == 1        

class TestTrackDeletion:

    def test_track_survives_within_max_age(self):
        t = SORTTracker(min_hits=1, max_age=3)
        t.update([make_detection()])
        #Miss two frames
        t.update([])
        t.update([])
        assert len(t.tracks) == 1
    
    def test_track_deleted_at_max_age_plus_one(self):
        t = SORTTracker(min_hits=1, max_age=1)
        t.update([make_detection()])
        #Miss a frame
        t.update([]) #time since last update = 1
        assert len(t.tracks) == 1
        t.update([]) #time since last update = 2 > max_age
        assert len(t.tracks) == 0
    
    def test_track_deleted_after_max_age(self):
        t = SORTTracker(min_hits=1, max_age=3)
        t.update([make_detection()])
        #Miss 3 frames
        for _ in range(4):
            t.update([])
        assert len(t.tracks) == 0
    
    def test_redetected_track_not_deleted(self):
        t = SORTTracker(min_hits=1, max_age=1)
        det = make_detection()
        t.update([det])
        t.update([]) #miss one frame
        t.update([det]) #Redetected - should reset time_since_last_update to zero
        assert len(t.tracks) == 1
        assert t.tracks[0].time_since_last_update == 0
        
class TestTrackIDAssignment:

    def test_first_track_id_is_zero(self):
        t = SORTTracker()
        t.update([make_detection()])
        assert t.tracks[0].track_id == 0

    def test_ids_are_unique_across_tracks(self):
        t = SORTTracker()
        t.update([make_detection(), far_detection()])
        ids = [track.track_id for track in t.tracks]
        assert len(ids) == len(set(ids))

    def test_ids_are_monotonically_increasing(self):
        t = SORTTracker()
        t.update([make_detection(), far_detection()])
        ids = [track.track_id for track in t.tracks]

        assert ids == sorted(ids)

    def test_new_track_after_deletion_gets_new_id(self):
        t = SORTTracker(max_age=1, min_hits=1)
        t.update([make_detection()])
        first_id = t.tracks[0].track_id
        t.update([]) #Miss one
        t.update([]) #Deleted
        t.update([make_detection()]) #New track born
        second_id = t.tracks[0].track_id

        assert second_id > first_id
    
    def test_next_id_increaments_at_birth(self):
        t = SORTTracker()
        t.update([make_detection()])
        assert t._next_id == 1
        t.update([far_detection()])
        assert t._next_id == 2

class TestMatchedTrackUpdate:

    def test_matched_tracks_hits_increaments(self):
        t = SORTTracker(max_age=5, min_hits=1)
        det = make_detection()
        t.update([det])
        assert t.tracks[0].hits == 1
        t.update([det])
        assert t.tracks[0].hits == 2
    
    def test_matched_track_time_since_last_update_resets(self):
        t = SORTTracker(max_age=5, min_hits=1)
        det = make_detection()
        t.update([det])
        t.update([]) #missed one frame
        assert t.tracks[0].time_since_last_update == 1
        t.update([det]) #time_since_last_update resets to zero
        assert t.tracks[0].time_since_last_update == 0
    

