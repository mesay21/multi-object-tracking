from __future__ import annotations

import numpy as np

def iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise IoU between two sets of bounding boxes.

    Args:
        boxes_a: Array of shape (N, 4) in [x1, y1, x2, y2] format.
        boxes_b: Array of shape (M, 4) in [x1, y1, x2, y2] format.
    
    Returns:
        IoU matrix of shape (N, M) where entry [i, j] is the IoU between 
        boxes_a[i] and boxes_b[j]. Values are in [0, 1]
    """
    if boxes_a.shape[0] == 0 or boxes_b.shape[0] == 0:
        return np.zeros(shape=(boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float64)
    
    #Broadcast boxes_a to (N, 1, 4) and boxes_b to (1, M, 4)
    boxes_a = np.expand_dims(boxes_a, axis=1)
    boxes_b = np.expand_dims(boxes_b, axis=0)

    #Compute intersection corners
    intersection_x1 = np.maximum(boxes_a[..., 0], boxes_b[..., 0])
    intersection_y1 = np.maximum(boxes_a[..., 1], boxes_b[..., 1])
    intersection_x2 = np.minimum(boxes_a[..., 2], boxes_b[..., 2])
    intersection_y2 = np.minimum(boxes_a[..., 3], boxes_b[..., 3])

    #Intersection area
    intersection_area = np.multiply(
        np.maximum(0.0, intersection_x2 - intersection_x1), 
        np.maximum(0.0, intersection_y2 - intersection_y1)
    )
    #Compute box areas
    area_a = np.multiply(
        boxes_a[..., 2] - boxes_a[..., 0], 
        boxes_a[..., 3] - boxes_a[..., 1]
    )
    area_b = np.multiply(
        boxes_b[..., 2] - boxes_b[..., 0],
        boxes_b[..., 3] - boxes_b[..., 1]
    )
    union_area = area_a + area_b - intersection_area

    return np.where(union_area > 0.0, intersection_area / union_area, 0.0)