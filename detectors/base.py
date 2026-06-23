from abc import ABC, abstractmethod

import numpy as np

from core.detection import Detection

class BaseDetector(ABC):
    """
    Abstract class for all detector wrappers.
    All detectors share the same contract:
        Input: BGR np.ndarray of shape (H, W, 3)
        Output: list[Detection] in pixel coordinate (x1, y1, x2, y2)
    Subclasses handle all internal processing (e.g normalization, color conversion, etc) and post processing (class filtering, format conversion, etc) 
    """
    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Takes BGR frame (H, W, 3), returns detections for that frame.
        Args:
            frame: BGR image as uint8 np.ndarray of shape (H, W, 3)
        Returns:
            List of Detection objects in [x1, y1, x2, y2] pixel coordinates filtered by confidence threshold and class ID.
        """
        ...
    
    @abstractmethod
    def warmup(self) -> None:
        """
        Optional: run a dummy forward to initalize CUDA kernels.
        Should be called once after initialization before the first inference call to avoid slow first frame.
        """
        ...