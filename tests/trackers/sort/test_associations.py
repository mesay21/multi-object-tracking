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
from unittest import result

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

class TestPerfectMatches:
    def test_identical_pairs_matched_correctly(self):
        tracks = [
            make_detection(0, 0, 10, 10),
            make_detection(20, 20, 30, 30)
        ]

        matched, unmatched_dets, unmatched_tracks = associate(tracks, tracks)

        assert len(matched) == len(tracks)
        assert (0, 0) in matched
        assert (1, 1) in matched

        assert not unmatched_dets
        assert not unmatched_tracks
    
    def test_matched_indices_are_valid(self):
        tracks = [
            make_detection(0, 0, 10, 10),
            make_detection(20, 20, 30, 30)
        ]
        matched, _, _ = associate(tracks, tracks)

        for track_idx, det_idx in matched:
            assert 0 <= track_idx < len(tracks)
            assert 0 <= det_idx < len(tracks)

class TestIoUThreshold:

    def test_match_rejected_below_threshold(self):
        """
        Matched detection below IoU threshold should be rejected.
        """
        matched, unmatched_dets, unmatched_tracks = associate(
            [make_detection(0, 0, 10, 10)],
            [make_detection(9, 0, 19, 10)],
            iou_threshold=0.3
        )

        assert not matched
        assert unmatched_dets == [0]
        assert unmatched_tracks == [0]

    def test_match_accepted_above_threshold(self):
        """
        Matched detection above IoU threshold should be accepted.
        """
        matched, _, _ = associate(
            [make_detection(0, 0, 10, 10)],
            [make_detection(0, 0, 10, 10)],
            iou_threshold=0.3
        )

        assert len(matched) == 1

    def test_match_accepted_at_threshold(self):
        """
        Matched detection at IoU threshold should be accepted.
        """
        matched, _, _ = associate(
            [make_detection(0, 0, 10, 10)],
            [make_detection(0, 0, 10, 5)],
            iou_threshold=0.5
        )

        assert len(matched) == 1  

    def test_non_overlapping_all_unmatched(self):
        """
        Matched detection above IoU threshold should be accepted.
        """
        matched, unmatched_dets, unmatched_tracks = associate(
            [make_detection(0, 0, 10, 10)],
            [make_detection(20, 20, 30, 30)],
            iou_threshold=0.3
        )

        assert not matched 
        assert set(unmatched_dets) == {0}
        assert set(unmatched_tracks) == {0}
     
class TestUnmatchedIndices:

    def test_unmatched_det_index_is_correct(self):
        """
        track[0] matches det[0], det[1] is far away (i.e unmatched)
        """   
        _, unmatched_dets, unmatched_tracks = associate(
            [make_detection(0, 0, 10, 10)],
            [make_detection(0, 0, 10, 10),
            make_detection(20, 20, 30, 30)]
        )
        assert unmatched_dets == [1]
        assert unmatched_tracks == []

    def test_unmatched_track_index_is_correct(self):
        """
        track[0] matches det[0], det[1] is far away (i.e unmatched)
        """   
        _, unmatched_dets, unmatched_tracks = associate(
            [make_detection(0, 0, 10, 10),
            make_detection(20, 20, 30, 30)],
            [make_detection(0, 0, 10, 10)]
        )
        assert unmatched_dets == []
        assert unmatched_tracks == [1]    
    
    def test_no_index_appears_in_both_matched_and_unmatched_detections(self):
        tracks = [
            make_detection(0, 0, 10, 10),
            make_detection(20, 20, 30, 30)
        ]
        dets = [
            make_detection(0, 0, 10, 10),
            make_detection(40, 40, 100, 100)
        ]

        matched, unmatched_dets, _ = associate(tracks, dets)

        matched_det_indices = {det_idx for _, det_idx in matched}

        assert matched_det_indices.isdisjoint(set(unmatched_dets))

    def test_no_index_appears_in_both_matched_and_unmatched_tracks(self):
        tracks = [
            make_detection(0, 0, 10, 10),
            make_detection(40, 40, 100, 100)

        ]
        dets = [
            make_detection(0, 0, 10, 10),
            make_detection(20, 20, 30, 30)
        ]

        matched, _, unmatched_tracks = associate(tracks, dets)

        matched_track_indices = {track_idx for track_idx, _ in matched}

        assert matched_track_indices.isdisjoint(set(unmatched_tracks))
    
    def test_all_indices_accounted_for(self):
        """
        Every detection box should appear exactly in one of matched or unmatched
        """
        tracks = [
            make_detection(0, 0, 10, 10),
            make_detection(20, 20, 30, 30)

        ]
        dets = [
            make_detection(0, 0, 10, 10),
            make_detection(20, 20, 30, 30)
        ]       
        matched, unmatched_dets, unmatched_tracks = associate(tracks, dets)

        all_det_indices = set(range(len(dets)))
        matched_det_indices = {det_idx for _, det_idx in matched}

        assert matched_det_indices | set(unmatched_dets) == all_det_indices

        all_track_indices = set(range(len(tracks)))
        matched_track_indices = {track_idx for track_idx, _ in matched}

        assert matched_track_indices | set(unmatched_tracks) == all_track_indices

class TestAsymmetricCounts:

    def test_more_detections_than_trackers(self):
        tracks = [make_detection(0, 0, 10, 10)]
        dets = [
            make_detection(0, 0, 10, 10),
            make_detection(20, 20, 30, 30),
            make_detection(40, 40, 60, 60)
        ]
        matched, unmatched_dets, unmatched_tracks = associate(tracks, dets)
        assert len(matched) == 1
        assert len(unmatched_dets) == 2
        assert not unmatched_tracks

    def test_more_tracks_than_detections(self):
        dets = [make_detection(0, 0, 10, 10)]
        tracks = [
            make_detection(0, 0, 10, 10),
            make_detection(20, 20, 30, 30),
            make_detection(40, 40, 60, 60)
        ]
        matched, unmatched_dets, unmatched_tracks = associate(tracks, dets)
        assert len(matched) == 1
        assert not unmatched_dets
        assert len(unmatched_tracks) == 2
    
    def test_single_track_multiple_detections(self):
        track = [make_detection(0, 0, 10, 10)]
        dets = [
            make_detection(0, 0, 10, 10), #IoU = 1
            make_detection(5, 5, 15, 15), #IoU = 0.14 
        ]

        matched, _, _ = associate(track, dets)

        assert len(matched) == 1
        assert matched[0] == (0, 0) #Must be matched to index 0

class TestReturnOrder:

    def test_return_order_is_correct(self):
        """
        Explicity verify return order is matched, unmatched_dets, and unmatched_tracks.
        A swap would cause tracker to use wrond indices
        """

        tracks = [
            make_detection(0, 0, 10, 10),
            make_detection(20, 20, 30, 30)
        ]

        dets = [
            make_detection(0, 0, 10, 10),
            make_detection(30, 30, 50, 50)
        ]
        matched, unmatched_dets, unmatched_tracks = associate(tracks, dets)

        #track[0] is assigned to det[0]
        assert (0, 0) in matched[0]
        #det[1] is unmatched

        assert 1 in unmatched_dets[1]
        #tracks[1] is unmatched

        assert 1 in unmatched_tracks[2]
