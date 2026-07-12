"""
Implements Faster-RCNN detector wrapper.

Uses torchvision's fasterrcnn_resnet50_fpn_v2 with COCO pretrained weights.
The models internal GeneralizedRCNNTrasform handles resizing and ImageNet normalization - no need to apply normalization manually. 

COCO class IDs:
    1 = person
    2 = bicycle
    3 = car
    ...
    0 = background (never returned)
"""

from __future__ import annotations

import numpy as np
import torch
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2
)
from loguru import logger

from core.detection import Detection
from detectors.base import BaseDetector

class FasterRCNNDetector(BaseDetector):
    """
    Torchvision Faster-RCNN wrapper.
    Args:
        conf_threshold: Minimum score to keep detection. Defaul 0.5.
        class_ids: COCO class IDs to keep (1 Indexed). Default [1] (Person).
        device: Pytorch device string. Deafult "cuda" with CPU fallback
    
    Example:
        detector = FasterRCNNDetector(conf_threshold = 0.5, class_ids = [1])
        detector.warmup()
        detections = detector.detect(frame) #frame is BGR np.ndarray
    """

    def __init__(
        self, 
        conf_threshold: float = 0.5,
        class_ids: list[int] | None = None,
        device: str | None = None
        ) -> None:
        self.conf_threshold = conf_threshold
        self.class_ids = set(class_ids) if class_ids is not None else {1}
        
        #Fall back to CPU if gpu is not available
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Loading Faster-RCNN ResNet-50 FPN v2 on {self.device} ...")

        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self._model = fasterrcnn_resnet50_fpn_v2(weights=weights)
        self._model.to(self.device)
        self._model.eval()

        logger.info("Faster-RCNN loaded successfully")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run Inference on a single BGR image.

        Args:
            frame: BGR uint8 np.ndarray of shape (H, W, 3)
        
        Returns:
            List of Detection objects filtered by conf_threshold and class_ids
        """
        tensor = self._preprocess(frame)

        with torch.no_grad():
            outputs = self._model([tensor])
        
        return self._postprocess(outputs[0])
        
    def warmup(self) -> None:
        """
        Run a dummy forward pass to initialize CUDA kernels (avoids delay in processing the first frame)
        Call once after class construction.
        """
        logger.info("Warming up Faster-RCNN...")
        dummy = torch.zeros(3, 640, 640, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            self._model([dummy])
        
        logger.info("Warmup complete")

        #-----------------------------------------------------------------------
        # Helpers
        #-----------------------------------------------------------------------

    def _preprocess(self, frame: np.ndarra) -> torch.Tensor:
        """
        Convert a BGR uint8 frame to a float32 RGB tensor in [0, 1].
        Torchvision's GeneralizedRCNNTrasform expects:
            - Shape: (3, H, W)
            - dtype: float32
            - Range: [0, 1]
            - Channel order: RGB
        Other preprocessing steps (e.g Normalization) are applied internally.

        Args:
            frame: BGR uint8 np.ndarray of shape (H, W, 3)
        
        Returns:
            Float32 torch tensor of shape (3, H, W) on self.device.
        """
        #BGR to RGB
        rgb = frame[..., ::-1].copy() #Copy makes in contiguous after flip
        #(H, W, 3) uint8 to (3, H, W) float32
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

        return tensor.to(self.device)
        
    def _postprocess(self, output: dict) -> list[Detection]:
        """
        Filter model outputs and convert to Detection objects.

        Args:
            output: Dict with keys boxes (N, 4), labels (N, ), and scores (N, ).
                    Boxes are in [x1, y1, x2, y2] absolute pixel coordinate.
        
        Returns:
            Filtered list of Detection objects.
        """

        boxes = output["boxes"].cpu().numpy() #(N, 4) float32
        labels = output["labels"].cpu().numpy() #(N, ) int64
        scores = output["scores"].cpu().numpy() #(N, ) float32

        detections: list[Detection] = []

        for box, label, score in zip(boxes, labels, scores):
            #Torchvision returns results sorted by score (Descending order)
            #If we are below the threshold we can break eary
            if score < self.conf_threshold:
                break
            
            if not int(label) in self.class_ids:
                continue
            
            x1, y1, x2, y2 = box.tolist()

            detections.append(
                Detection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    score=float(score),
                    class_id=int(label)
                )
            )
        return detections
        
    def __repr__(self) -> str:
        return (
            f"FasterRCNNDetector("
            f"conf_threshold={self.conf_threshold}, "
            f"class_ids={sorted(self.class_ids)}, "
            f"device={self.device})"
        )

