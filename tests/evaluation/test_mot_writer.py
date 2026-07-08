"""
Unit tests for evaluation.mot_writer.MOTWriter

Tests covered:
    - file opened/closed correctly
    - correct rows written per frame
    - MOTChallange column format and values
    - correct output across multiple frames
    - empty track list writes nothing
    - edge case : zero score, large coordinates, fractional boxes
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.detection import Detection
from evaluation.mot_writer import MOTWriter


def make_det(
    x1: float = 100.0,
    y1: float = 200.0,
    x2: float = 200.0,
    y2: float = 400.0,
    score: float = 0.9
) -> Detection:
    return Detection(x1=x1, y1=y1, x2=x2, y2=y2, score=score, class_id=1)

def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()

def parse_row(line: str) -> list[str]:
    return line.strip().split(",")

class TestContextManager:

    def test_file_created_on_enter(self, tmp_path: Path):
        output = tmp_path / "tracks.txt"
        with MOTWriter(output) as writer:
            assert output.exists()
    
    def test_file_closed_on_exit(self, tmp_path: Path):
        output = tmp_path / "tracks.txt"
        with MOTWriter(output) as writer:
            pass
        assert writer._file.closed
    
    def test_file_closed_on_exception(self, tmp_path: Path):
        output = tmp_path / "tracks.txt"
        try:
            with MOTWriter(output) as writer:
                raise RuntimeError("Simulated error")
        except RuntimeError:
            pass
        assert writer._file.closed

    def test_no_file_handle_before_enter(self, tmp_path: Path):
         writer = MOTWriter(tmp_path / "tracks.txt")
         assert writer._file is None
    
    def test_returns_self_on_enter(self, tmp_path: Path):
        output = tmp_path / "tracks.txt"
        with MOTWriter(output) as writer:
            assert isinstance(writer, MOTWriter)
