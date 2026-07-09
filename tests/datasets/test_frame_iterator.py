"""
Unit tests for datasets.frame_iterator.FrameIterator.
- Synthetic video files are created using Cv2.VideoWriter (no need for external data)
- Using unittest.mock.MagicMock to mock detector. (no need to load Faster-RCNN model)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from core.detection import Detection
from datasets.frame_iterator import Frame, FrameIterator

HEIGHT, WIDTH = 240, 320
FPS = 10
NUM_FRAMES = 8

def make_detection(score: float=0.9) -> Detection:
    return Detection(x1=10.0, y1=20.0, x2=50.0, y2=100.0, score=score, class_id=1)


def mock_detector(detections_per_frame: list[Detection] | None = None) -> MagicMock:
    """
    Run a mock BaseDetector whose detect() returns a fixed list.
    """
    detector = MagicMock()
    dets = detections_per_frame if detections_per_frame is not None else [make_detection()]
    detector.detect.return_value = dets

    return detector

@pytest.fixture()
def synthetic_video(tmp_path: Path) -> Path:
    """
    Write a short synthetic video to temp path and return its path.
    """
    video_path = tmp_path / "test_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, FPS, (WIDTH, HEIGHT))

    rng = np.random.default_rng(seed=42)
    for _ in range(NUM_FRAMES):
        frame = rng.integers(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8)
        writer.write(frame)
    
    writer.release()
    assert video_path.exists(), "VideoWriter failed to create file."

    return video_path

@pytest.fixture()
def non_video_file(tmp_path: Path) -> Path:
    """
    Plain text file - not a valid video file.
    """
    p = tmp_path / "not_video.txt"
    p.write_text("Hello")

    return p


class TestFrameIteratorValidation:
    def test_missing_source_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            FrameIterator(
                source=tmp_path / "nonexistent.mp4",
                detector=mock_detector()
            )
    
class TestFrameIteratorIteration:
    def test_yields_correct_number_of_frames(self, synthetic_video: Path):
        detector = mock_detector()
        iterator = FrameIterator(source=synthetic_video, detector=detector)
        frames = list(iterator)
        assert len(frames) == NUM_FRAMES
    
    def test_yields_frame_dataclass(self, synthetic_video: Path):
        iterator = FrameIterator(source=synthetic_video, detector=mock_detector())
        frame = next(iter(iterator))
        assert isinstance(frame, Frame)
    
    def test_one_based_frame_indices(self, synthetic_video: Path):
        iterator = FrameIterator(source=synthetic_video, detector=mock_detector())
        indices = [f.index for f in iterator]
        assert indices == list(range(1, NUM_FRAMES + 1))
    
    def test_frame_image_shape(self, synthetic_video: Path):
        iterator = FrameIterator(source=synthetic_video, detector=mock_detector())
        frame = next(iter(iterator))
        assert frame.image.ndim == 3
        assert frame.image.shape[2] == 3 #BGR image
    
    def test_frame_image_dtype(self, synthetic_video: Path):
        iterator = FrameIterator(source=synthetic_video, detector=mock_detector())
        frame = next(iter(iterator))
        assert frame.image.dtype == np.uint8

    def test_detections_passed_through(self, synthetic_video: Path):
        expected = [make_detection(score=0.95), make_detection(score=0.8)]
        detector = mock_detector(detections_per_frame=expected)
        iterator = FrameIterator(source=synthetic_video, detector=detector)
        frame = next(iter(iterator))
        assert frame.detections == expected
    
    def test_detector_called_once_per_frame(self, synthetic_video: Path):
        detector = mock_detector()
        iterator = FrameIterator(source=synthetic_video, detector=detector)
        list(iterator)
        assert detector.detect.call_count == NUM_FRAMES
    
    def test_with_multiple_detections_per_frame(self, synthetic_video: Path):
        dets = [make_detection(score=0.95), 
                make_detection(score=0.7), 
                make_detection(score=0.5)]
        detector = mock_detector(detections_per_frame=dets)
        iterator = FrameIterator(source=synthetic_video, detector=detector)
        frame = next(iter(iterator))
        assert len(frame.detections) == 3
    
    def test_empy_detection_per_frame(self, synthetic_video: Path):
        detector = mock_detector(detections_per_frame=[])
        iterator = FrameIterator(source=synthetic_video, detector=detector)
        for frame in iterator:
            assert frame.detections == []

class TestFrameIteratorConfFilter:
    def test_detections_above_threshold(self, synthetic_video: Path):
        detections = [make_detection(score=0.95), make_detection(score=0.8)]
        detector = mock_detector(detections_per_frame=detections)
        iterator = FrameIterator(source=synthetic_video,
                                detector=detector,
                                conf_threshold=0.9)
        frame = next(iter(iterator))
        assert len(frame.detections) == 1
        assert frame.detections[0].score == 0.95

    def test_all_detections_filtered_yields_empty_list(self, synthetic_video: Path):
        detections = [make_detection(score=0.5), make_detection(score=0.8)]
        detector = mock_detector(detections_per_frame=detections)
        iterator = FrameIterator(source=synthetic_video,
                                detector=detector,
                                conf_threshold=0.9)
        frame = next(iter(iterator))
        assert frame.detections == []

    def test_zero_theresholds_keeps_all(self, synthetic_video: Path): 
        detections = [make_detection(score=0.5), make_detection(score=0.8)]      
        detector = mock_detector(detections_per_frame=detections)
        iterator = FrameIterator(source=synthetic_video,
                                detector=detector,
                                conf_threshold=0.0)
        frame = next(iter(iterator))
        assert len(frame.detections) == len(detections)
    
    def test_threshold_at_exact_score_keep_detection(self, synthetic_video: Path):
        detections = [make_detection(score=0.5), make_detection(score=0.8)]      
        detector = mock_detector(detections_per_frame=detections)
        iterator = FrameIterator(source=synthetic_video,
                                detector=detector,
                                conf_threshold=0.8)
        frame = next(iter(iterator))
        assert len(frame.detections) == 1
        assert frame.detections[0].score == 0.8

class TestFrameIteratorLen:
    def test_len_matches_frame_count(self, synthetic_video: Path):
        iterator = FrameIterator(source=synthetic_video,
                                detector=mock_detector())
        assert abs(len(iterator) - NUM_FRAMES) <= 1

class TestFrameIteratorRelease:
    def test_capture_released_on_early_break(self, synthetic_video: Path):
        """
        VideoCapture must be released even if caller breaks mid-iteration. 
        """        
        released = []

        original_release = cv2.VideoCapture.release

        def track_release(self_cap):
            released.append(True)
            return original_release(self_cap)
        
        iterator = FrameIterator(source=synthetic_video,
                                detector=mock_detector())
        
        with patch.object(cv2.VideoCapture, "release", track_release):
            for i, _ in enumerate(iterator):
                if i == 1:
                    break
        
        assert len(released) >= 1, "VideoCapture.release() was never called"
    
    def test_released_after_full_iteration(self, synthetic_video: Path):
        released = []

        original_release = cv2.VideoCapture.release

        def track_release(self_cap):
            released.append(True)
            return original_release(self_cap)
        
        iterator = FrameIterator(source=synthetic_video,
                                detector=mock_detector())
        
        with patch.object(cv2.VideoCapture, "release", track_release):
            list(iterator)
        
        assert len(released) >= 1, "VideoCapture.release() was never called"
        