"""
Unit tests for visualization.renderer

Tests covered:
    - Boxes drawn on a copy and original frame is not modified
    - Distinct colors per track
    - Box thickness scales with resolution
    - Context manager for writting frames and resource is released on exit
    - Visualization exits on q-press
"""

from __future__ import annotations

from pathlib import Path
from socket import IP_DEFAULT_MULTICAST_TTL
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from core.detection import Detection
from tests.datasets.test_frame_iterator import HEIGHT
from visualization.renderer import(
    VideoWriter,
    _box_thickness,
    _track_color,
    draw_frame,
    display_frame
)

HEIGHT, WIDTH = 480, 640
FPS = 30

def blank_frame(h: int = HEIGHT, w: int = WIDTH) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)

def make_det(
    x1: float = 50.0,
    y1: float = 100.0,
    x2: float = 150.0,
    y2: float = 300.0,
    score: float = 0.85

) -> Detection:
    return Detection(x1=x1, y1=y1, x2=x2, y2=y2, score=score, class_id=1)

def make_tracks(n: int = 1) -> list[tuple[Detection, int]]:
    #Sample n top left coordinates
    rng = np.random.default_rng(seed=0)
    left = rng.integers(low=0, high=WIDTH - 100, size=(n, ))
    top = rng.integers(low=0, high=HEIGHT - 100, size=(n, ))
    #Create detections
    tracks = []
    for idx, (x1, y1) in enumerate(zip(left, top)):
        tracks.append((make_det(x1, y1, x1+50, y1+50), idx))
    return tracks

@pytest.fixture()
def synthetic_video(tmp_path: Path) -> Path:
    path = tmp_path / "test.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_path), fourcc, FPS, (WIDTH, HEIGHT))
    rng = np.random.default_rng(seed=0)
    for _ in range(10):
        writer.write(rng.integers(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8))
    writer.release()

    return path

class TestDrawFrame:

    def test_returns_ndarray(self):
        result = draw_frame(blank_frame(), make_tracks())
        assert isinstance(result, np.ndarray)

    def output_shape_matches_input(self):
        frame = blank_frame()
        result = draw_frame(frame, make_tracks())
        assert frame.shape == result.shape

    def test_output_dtype_is_uint8(self):
        result = draw_frame(blank_frame(), make_tracks())
        assert result.dtype == np.uint8

    def test_original_frame_not_modified(self):
        frame = blank_frame()
        original = frame.copy()
        draw_frame(frame, make_tracks())
        np.testing.assert_array_equal(frame, original)
    
    def test_empty_track_returns_copy_of_frame(self):
        frame = blank_frame()
        result = draw_frame(frame, [])
        np.testing.assert_array_equal(result, frame)
        assert result is not frame #Result must be a copy
    
    def test_annotated_frame_differs_from_blank(self):
        frame = blank_frame()
        result = draw_frame(frame, make_tracks())
        assert not np.array_equal(result, frame)
    
    def test_multiple_tracks_all_drawn(self):
        frame = blank_frame()
        result_one = draw_frame(frame, make_tracks(1))
        result_two = draw_frame(frame, make_tracks(2))
        #More tracks more pixels changed
        diff_one = np.sum(result_one != frame)
        diff_two = np.sum(result_two != frame)
        assert diff_two >= diff_one
    
    def test_box_coordinates_casted_to_int(self):
        """
        Fractional coordinates should not raise error - verifies int() cast.
        """
        frame = blank_frame()
        det = make_det(x1=10.7, y1=20.3, x2=50.9, y2=100.1)
        draw_frame(frame, [(det, 0)]) #Should not aise error

class TestTrackColor:

    def test_returns_tuple_of_three(self):
        color = _track_color(0)
        assert isinstance(color, tuple)
        assert len(color) == 3
    
    def test_values_in_valid_range(self):
        for track_id in range(200):
            color = _track_color(track_id)
            assert all(50 <= v <= 255 for v in color)
    
    def test_same_id_returns_same_color(self):
        assert _track_color(7) == _track_color(7)
    
    def test_diffrent_ids_different_colors(self):
        colors = {_track_color(i) for i in range(10)}
        assert len(colors) > 1

class TestBoxThickness:
    
    def test_returns_int(self):
        assert isinstance(_box_thickness(blank_frame()), int)
    
    def test_default_thickness_is_two(self):
        tiny = blank_frame(100, 100)
        assert _box_thickness(tiny) >= 2
    def test_4k_frame_thickness_above_minimum(self):
        frame_4k = blank_frame(2160, 3840)
        assert _box_thickness(frame_4k) > 2