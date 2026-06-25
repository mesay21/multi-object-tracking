"""
Unit tests for core.geometry.iou_batch.
Tests included
    - Output shape should be (N, M) all the time
    - Handles empty inputs
    - Verify using Known values
    - Boundary conditions
    - Numerical stability
    - Boradcasting
"""

from __future__ import annotations
from unittest import result

import numpy as np
import pytest

from core.geometry import iou_batch


ATOL = 1e-6

def boxes(*args) -> np.ndarray:
    """
    Build a (N, 4) array from row tuples.
    """
    return np.array(args, dtype=np.float64)

class TestOutputShape:

    def test_square_output(self):
        a = boxes([0, 0, 10, 10], [20, 20, 30, 30])
        b = boxes([0, 0, 10, 10], [20, 20, 30, 30])

        result = iou_batch(a, b)

        assert result.shape == (2, 2)
    
    def test_rectangular_output_n_gt_m(self):
        a = boxes(
            [0, 0, 10, 10], 
            [20, 20, 30, 30],
            [40, 40, 50, 50]
        )
        b = boxes([0, 0, 10, 10])
        result = iou_batch(a, b)

        assert result.shape == (3, 1)
    
    def test_rectangular_output_m_gt_n(self):
        a = boxes([0, 0, 10, 10])
        b = boxes(
            [0, 0, 10, 10], 
            [20, 20, 30, 30],
            [40, 40, 50, 50]
        )
        result = iou_batch(a, b)

        assert result.shape == (1, 3)
    
    def test_single_pair(self):
        a = boxes([0, 0, 10, 10])
        b = boxes([0, 0, 10, 10])
        result = iou_batch(a, b)
        assert result.shape == (1, 1)
    
    def test_output_dtype_float(self):
        a = boxes([0, 0, 10, 10])
        b = boxes([0, 0, 10, 10])
        result = iou_batch(a, b)
        assert np.issubdtype(result.dtype, np.floating)      

class TestEmptyInputs:

    def test_empty_a_returns_zero_matrix(self):
        a = np.zeros((0, 4), dtype=np.float64)
        b = boxes(
            [0, 0, 10, 10], 
            [20, 20, 30, 30],
            [40, 40, 50, 50]
        )
        result = iou_batch(a, b) 
        assert result.shape == (0, 3)
    
    def test_empty_b_returns_zero_matrix(self):
        a = boxes(
            [0, 0, 10, 10], 
            [20, 20, 30, 30],
            [40, 40, 50, 50]
        )
        b = np.zeros((0, 4), dtype=np.float64)
        result = iou_batch(a, b) 
        assert result.shape == (3, 0)

    def test_both_empty_returns_zero_matrix(self):
        a = np.zeros((0, 4), dtype=np.float64)
        b = np.zeros((0, 4), dtype=np.float64)

        result = iou_batch(a, b)
        assert result.shape == (0, 0)

    def test_empty_result_contains_no_nan(self):
        a = np.zeros((0, 4), dtype=np.float64)
        b = boxes([0, 0, 10, 10])
        result = iou_batch(a, b)
        assert not np.any(np.isnan(result))    

                
        