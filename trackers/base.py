"""
Base class for all trackers.
"""
from abc import ABC, abstractmethod
from typing import List

from core.detection import Detection

class BaseTracker(ABC):
    """
    Base class for all trackers.
    """
    @abstractmethod
    def update(self, detections: List[Detection]) -> List[tuple[Detection, int]]:
        """
        Process one frame of detections

        Args:
            detections: List of detections objects from detector.
        
        Returns:
            List of (box, track_id) pair of confirmed tracks only.
        """
        ...