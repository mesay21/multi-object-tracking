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

class TestKnownValues:

    def test_identical_box_iou_is_one(self):
        a = boxes([10, 10, 50, 80])
        result = iou_batch(a, a)
        assert pytest.approx(result[0, 0], abs=ATOL) == 1.0
    
    def test_non_overlapping_box_iou_is_zero(self):
        a = boxes([10, 10, 50, 80])
        b = boxes([60, 90, 100, 120])    

        result = iou_batch(a, b)
        assert pytest.approx(result[0, 0], abs=ATOL) == 0.0

    def test_half_overlap_horizontal(self):
        #a = [0, 0, 10, 10], area = 100
        #b = [5, 0, 15, 10], area = 100
        # intersection: [5, 0, 10, 10] = 50
        # union = 100 + 100 - 50 = 150
        # IoU = 50/150 = 1/3
        a = boxes([0, 0, 10, 10])
        b = boxes([5, 0, 15, 10])    
        result = iou_batch(a, b)
        assert  pytest.approx(result[0, 0], abs=ATOL) == 1.0/3.0
    
    def test_quarter_overlap_horizontal(self):
        #a = [0, 0, 10, 10], area = 100
        #b = [5, 5, 15, 15], area = 100
        # intersection: [5, 5, 10, 10] = 25
        # union = 100 + 100 - 25 = 175
        # IoU = 25/175 = 1/7
        a = boxes([0, 0, 10, 10])
        b = boxes([5, 5, 15, 15])    
        result = iou_batch(a, b)
        assert  pytest.approx(result[0, 0], abs=ATOL) == 25.0/175.0
    
    def test_contained_box(self):
        """
        b is fully contained in a
        a: [0, 0, 100, 100], area = 10000
        b: [25, 25, 75, 75], area = 2500
        intersection: [25, 25, 75, 75], area = 2500
        union: 10000 + 2500 - 2500 = 10000
        IoU: 2500/10000 = 0.25
        """
        a = boxes([0, 0, 100, 100])
        b = boxes([25, 25, 75, 75])    
        result = iou_batch(a, b)
        assert  pytest.approx(result[0, 0], abs=ATOL) == 0.25   

    def test_with_known_values(self):
        """
        a[0] == b[0], IoU = 1
        a[0] == b[1], IoU = 0
        a[1] == b[0], IoU = 0
        a[1] == b[1], IoU = 1
        """ 
        a = boxes(
            [0, 0, 10, 10],
            [20, 20, 30, 30]
        )
        b = boxes(
            [0, 0, 10, 10],
            [20, 20, 30, 30]
        )    
        result = iou_batch(a, b)
        expected = np.array([[1.0, 0.0],[0.0, 1.0]])
        np.testing.assert_allclose(result, expected, atol=ATOL)
    
    def test_touching_edge_iou_is_zero(self): 
        """
        Boxes share only an edge - intersection is zero
        """      
        a = boxes([0, 0, 10, 10])
        b = boxes([10, 0, 20, 10])
        result = iou_batch(a, b)
        assert pytest.approx(result[0, 0], abs=ATOL) == 0.0

class TestSymmetry:

    def test_iou_is_symmetric(self):
        a = boxes(
            [0, 0, 10, 10],
            [5, 5, 20, 20]
        ) 
        b = boxes(
            [3, 3, 15, 15],
            [0, 0, 10, 10],
            [25, 25, 40, 40]
        )
        result_ab = iou_batch(a, b)
        result_ba = iou_batch(b, a)

        np.testing.assert_allclose(result_ab, result_ba.T, atol=ATOL)  

class TestBoundaryValues:

    def test_all_values_in_unit_interval(self):
        rng = np.random.default_rng(seed=0)
        #Generate valid boxes: x1 < x2, y1 < y2
        raw = rng.uniform(0, 100, size=(20, 4))
        a = np.column_stack(
            [
                np.minimum(raw[:10, 0], raw[:10, 2]),
                np.minimum(raw[:10, 1], raw[:10, 3]),
                np.maximum(raw[:10, 0], raw[:10, 2]),
                np.maximum(raw[:10, 1], raw[:10, 3])
            ]
        )    
        b = np.column_stack(
            [
                np.minimum(raw[10:, 0], raw[10:, 2]),
                np.minimum(raw[10:, 1], raw[10:, 3]),
                np.maximum(raw[10:, 0], raw[10:, 2]),
                np.maximum(raw[10:, 1], raw[10:, 3])
            ]
        )

        result = iou_batch(a, b)  
        assert np.all(result >= 0.0), "IoU must be >= 0"
        assert np.all(result <= 1.0), "IoU must be <= 1"

    def test_no_nan_values(self):
        a = boxes(
            [0, 0, 10, 10],
            [5, 5, 20, 20]
        ) 
        b = boxes(
            [3, 3, 15, 15],
            [0, 0, 10, 10],
            [25, 25, 40, 40]
        )

        result = iou_batch(a, b)
        assert not np.any(np.isnan(result))       
    
    def test_no_inf_values(self):
        a = boxes(
            [0, 0, 10, 10],
            [5, 5, 20, 20]
        ) 
        b = boxes(
            [3, 3, 15, 15],
            [0, 0, 10, 10],
            [25, 25, 40, 40]
        )

        result = iou_batch(a, b)
        assert not np.any(np.isinf(result))  

class TestNumericalStability:

    def test_zero_area_box_returns_zero_iou(self):
        a = boxes([10, 10, 10, 10])
        b = boxes([5, 5, 15, 15])
        c = boxes([10, 10, 10, 10])
        result_ab = iou_batch(a, b)
        result_bc = iou_batch(b, c)
        assert pytest.approx(result_ab[0, 0], abs=ATOL) == 0.0
        assert pytest.approx(result_bc[0, 0], abs=ATOL) == 0.0

    def test_large_coordinate_overflow(self):
        a = boxes([0, 0, 3840, 2160])
        b = boxes([1920, 0, 3840, 2160])
        result = iou_batch(a, b)
        assert np.isfinite(result[0, 0])
        assert result[0, 0] > 0

    def test_float32_input_accepted(self):  
        a = boxes([0, 0, 10, 10]).astype(np.float32)
        b = boxes([5, 5, 15, 15]).astype(np.float32)
        result = iou_batch(a, b)
        assert np.isfinite(result[0, 0])
        assert result[0, 0] > 0

    def test_small_overlap_not_negative(self):
        #1-pixel overlap
        a = boxes([0, 0, 10, 10])
        b = boxes([9, 9, 20, 20])
        result = iou_batch(a, b)
        assert result[0, 0] >= 0.0 

    def test_negative_coordinates_handeled(self):
        a = boxes([-10, -10, 10, 10])
        b = boxes([-5, -5, 5, 5])
        result = iou_batch(a, b)
        assert result[0, 0] > 0.0  