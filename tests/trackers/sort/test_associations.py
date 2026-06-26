"""
Unit test for trackers.sort.association.associate.

Tests covered:
    - Empty tracker/ detection input is handeled correctly
    - Return type and structure is correct
    - 1:1 matching with high IoU
    - Matches accepted/rejected at threshold boundary
    - Unmateched detections and tracks are correct
    - Correct behaviour with more trackers or detections and vice versa
    - Hungarian finds global optimal assignment
    - Return order is correct
"""

from __future__ import annotations

import numpy as np
import pytest

from core.detection import Detection
from trackers.sort.association import associate

def make_detection(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    score: float = 0.9 
) -> Detection:
    return Detection(x1=x1, y1=y1, x2=x2, y2=y2, score=score, class_id=1)

class TestEmptyInput:

    def test_empty_trackers(self):
        """
        Returns all detections unmatched
        """
        dets = [
            make_detection(0, 0, 10, 10), 
            make_detection(20, 20, 30, 30)
        ]
        matched, unmateched_det, unmatched_tracks = associate([], dets)
        
        assert not matched
        assert unmateched_det == [0, 1]
        assert not unmatched_tracks

    def test_empty_detections(self):
        """
        Returns all detections unmatched
        """
        tracks = [
            make_detection(0, 0, 10, 10), 
            make_detection(20, 20, 30, 30)
        ]
        matched, unmateched_det, unmatched_tracks = associate(tracks, [])
        
        assert not matched
        assert not unmateched_det
        assert unmatched_tracks == [0, 1]
    
    def test_both_empty(self):
        """
        Both detection and trackers empty should return all empty
        """
        matched, unmatched_det, unmatched_tracks = associate([], [])

        assert not matched
        assert not unmatched_det
        assert not unmatched_tracks

class TestReturnTypes:

    def test_returns_three_values(self):
        result = associate(
            [make_detection(0, 0, 10, 10)],
            [make_detection(20, 20, 30, 30)]
        )

        assert len(result) == 3
    
    def test_matched_is_list_of_tuples(self):
        matched, _, _ = associate(
            [make_detection(0, 0, 10, 10)],
            [make_detection(0, 0, 30, 30)]
        )

        assert isinstance(matched, list)
        if matched:
            assert isinstance(matched[0], tuple)
            assert len(matched[0]) == 2
    
    def test_unmatched_det_is_list_of_ints(self):
        _, unmatched_dets, _ = associate(
            [],
            [make_detection(0, 0, 10, 10)],
        )
        assert isinstance(unmatched_dets, list)
        assert all(isinstance(i, int) for i in unmatched_dets)

    def test_unmatched_trackers_is_list_of_ints(self):
        _, _, unmatched_tracks = associate(
            [make_detection(0, 0, 10, 10)],
            []
        )
        assert isinstance(unmatched_tracks, list)
        assert all(isinstance(i, int) for i in unmatched_tracks)