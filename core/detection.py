from __feature__ import annotations

from dataclasses import dataclass

import numpy as np

@dataclass
class Detection:
    """
    A single bounding box detection from a detector.
    All coordinates are in absolute pixel space referenced to the original frame (before frame is resized internally)

    Attributes:
        x1: left edge of the bbox (in pixels)
        y1: top edge of the bbox (in pixels)
        x2: right edge of the bbox (in pixels)
        y2: bottom edge of the bbox (in pixels)
        score: Detector confidence [0, 1]
        class_id: Integer class label
    """

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int

    #---------------------------------------------------------------------------
    # Format convertors
    #---------------------------------------------------------------------------

    def to_xyxy(self) -> np.ndarray:
        """
        Return [x1, y1, x2, y2] as a float32 array of shape (4, ).
        """
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)
    
    def to_xywh(self) -> np.ndarray:
        """
        Return [x1, y1, w, h] as a float32 array of shape (4, ).
        w: bbox width
        h: bbox height
        """
        w = self.x2 - self.x1
        h = self.y2 - self.y1

        return np.array([self.x1, self.y1, w, h], dtype=np.float32)
    
    def to_cxcysr(self) -> np.ndarray:
        """
        Return [cx, cy, s, r] as a float32 array of shape (4, ).
        cx: horizontal center
        cy: vertical center
        s: scale (scale = w*h)
        r: aspect ratio (w/h)
        """
        w = self.x2 - self.x1
        h = self.y2 - self.y1
        cx = self.x1 + w/2
        cy = self.y1 + h/2
        s = w * h
        r = w/h

        return np.array([cx, cy, s, r], dtype=np.float32)

    #---------------------------------------------------------------------------
    # Class method constructs from other formats
    #---------------------------------------------------------------------------

    @classmethod
    def from_xyxy(
        cls,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        score: float = 1.0,
        class_id: int = 0

    ) -> Detection:
        """
        Construct from raw [x1, y1, x2, y2] values.
        """
        return cls(x1=x1, y1=y1, x2=x2, y2=y2, score=score, class_id=class_id)

    @classmethod
    def from_cxcysr(
        cls,
        cx: float,
        cy: float,
        s: float,
        r: float,
        score: float = 1.0,
        class_id: int = 0
    ) -> Detection:
        """
        Construct from raw [x1, y1, x2, y2] values from cxcysr format.
        Args:
            cx: horizontal center
            cy: vertical center
            s: scale (area = w*h)
            r: aspect ratio (w/h)
        """
        w = np.sqrt(s*r)
        h = s / w if w > 0 else  0.0
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0

        return cls(x1=x1, y1=y1, x2=x2, y2=y2, score=score, class_id=class_id)
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    def __repr__(self) -> str:
        return (
            f"Detection(x1={self.x1:.1f}, y1={self.y1:.1f}, "
            f"x2={self.x2:.1f}, y2={self.y2:.1f}, "
            f"score={self.score:.3f}, class_id={self.class_id}"
        )
    