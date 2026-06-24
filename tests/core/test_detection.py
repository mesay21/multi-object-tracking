"""
Unit tests for core.detection.Detection
Coverage:
    - to_xyxy, to_xywh, and to_cxcysr format convertors
    - from_xyxy, from_cxcysr constructors
    - xyxy to cxcysr to xyxy check
    - width, height, and area properties
    - Edge cases: square, wide, tall, near-zero boxes
"""

import numpy as np
import pytest

from core.detection import Detection

ATOL = 1e-5 #Tolerance for float comparisons

def make_detection(x1=10.0, y1=20.0, x2=50.0, y2=100.0, score=0.8, class_id=1):
    return Detection(x1=x1, y1=y1, x2=x2, y2=y2, score=score, class_id=class_id)

class TestToXyxy:

    def test_values(self):
        det = make_detection(x1=10, y1=20, x2=50, y2=100)
        result = det.to_xyxy()
        np.testing.assert_allclose(result, [10.0, 20.0, 50.0, 100.0], atol=ATOL)
    
    def test_dtype(self):
        result = make_detection().to_xyxy()
        assert result.dtype == np.float32
    
    def test_shape(self):
        result = make_detection().to_xyxy()
        assert result.shape == (4,)

class TestXywh:

    def test_values(self):
        det = make_detection(x1=10, y1=20, x2=50, y2=100)
        result = det.to_xywh()
        np.testing.assert_allclose(result, [10.0, 20.0, 40.0, 80.0], atol=ATOL)
    
    def test_dtype(self):
        result = make_detection().to_xywh()
        assert result.dtype == np.float32
    
    def test_shape(self):
        result = make_detection().to_xywh()
        assert result.shape == (4,)
    
    def test_square_box(self):
        det = make_detection(x1=0, y1=0, x2=64, y2=64)
        result = det.to_xywh()
        np.testing.assert_allclose(result, [0.0, 0.0, 64.0, 64.0], atol=ATOL)

class TestCxcysr:

    def test_values(self):
        det = make_detection(x1=10, y1=20, x2=50, y2=100)
        result = det.to_cxcysr()
        #cx=30, cy=60, s=3200, r=0.5
        np.testing.assert_allclose(result, [30.0, 60.0, 3200.0, 0.5], atol=ATOL)
    
    def test_square_box(self):
        det = make_detection(x1=0, y1=0, x2=64, y2=64)
        result = det.to_cxcysr()
        #w=h=64, s=4096, r=1
        np.testing.assert_allclose(result, [32.0, 32.0, 4096.0, 1.0], atol=ATOL)
    
    def test_dtype(self):
        result = make_detection().to_cxcysr()
        assert result.dtype == np.float32
    
    def test_shape(self):
        result = make_detection().to_cxcysr()
        assert result.shape == (4,)

class TestFromCxcysr:
    def test_from_cxcysr(self):
        #cx=30, cy=60, s=3200, r=0.5
        #w=40, h=80
        #x1=10, y1=20, x2=50, y2=100

        det = Detection.from_cxcysr(cx=30, cy=60, s=3200, r=0.5)
        assert pytest.approx(det.x1, abs=ATOL) == 10.0
        assert pytest.approx(det.y1, abs=ATOL) == 20.0
        assert pytest.approx(det.x2, abs=ATOL) == 50.0
        assert pytest.approx(det.y2, abs=ATOL) == 100.0

    
    def test_defaults(self):
        det = Detection.from_cxcysr(cx=0.0, cy=0.0, s=100.0, r=1.0)
        assert det.score == 1.0
        assert det.class_id == 0

class TestRoundTrip:
    """
    Test the conversion from xyxy to cxcysr back to xyxy has the same result.
    """
    @pytest.mark.parametrize(
        "x1, y1, x2, y2",
        [

            (10.0, 20.0, 50.0, 100.0),
            (0.0, 0.0, 64.0, 64.0),
            (100.0, 200.0, 150.0, 400.0),
            (0.5, 0.5, 1.5, 1.5) #sub-pixel box
        ],
    )

    def test_xyxy_cxcysr_xyxy(self, x1, y1, x2, y2):
        original = Detection(x1=x1, y1=y1, x2=x2, y2=y2, score=0.8, class_id=1)
        cx, cy, s, r = original.to_cxcysr()
        recovered = Detection.from_cxcysr(cx=cx, cy=cy, s=s, r=r)
        assert pytest.approx(recovered.x1, abs=ATOL) == original.x1
        assert pytest.approx(recovered.y1, abs=ATOL) == original.y1
        assert pytest.approx(recovered.x2, abs=ATOL) == original.x2
        assert pytest.approx(recovered.y2, abs=ATOL) == original.y2

class TestProperties:

    def test_width(self):
        det = make_detection(x1=10, y1=20, x2=50, y2=100)
        assert pytest.approx(det.width, abs=ATOL) == 40.0
    
    def test_height(self):
        det = make_detection(x1=10, y1=20, x2=50, y2=100)
        assert pytest.approx(det.height, abs=ATOL) == 80.0

    def test_area(self):
        det = make_detection(x1=10, y1=20, x2=50, y2=100)
        assert pytest.approx(det.area, abs=ATOL) == 3200.0

class TestEdgeCases:

    def test_zero_heigh_box(self):
        """
        Test to_cxcysr handles h=0 case without error
        """
        det = Detection.from_cxcysr(cx=25.0, cy=25.0, s=0.0, r=1.0)
        assert not any(
            v != v for v in [det.x1, det.y1, det.x2, det.y2]
        ) #NaN check
    
    def test_single_pixel_box(self):
        det = make_detection(x1=5.0, y1=5.0, x2=6.0, y2=6.0)
        assert pytest.approx(det.area, abs=ATOL) == 1.0
        cx, cy, s, r = det.to_cxcysr()
        assert pytest.approx(cx, abs=ATOL) == 5.5
        assert pytest.approx(cy, abs=ATOL) == 5.5
        assert pytest.approx(s, abs=ATOL) == 1.0
        assert pytest.approx(r, abs=ATOL) == 1.0
    
    def test_large_box(self):
        """
        Check no overflow for large bbox. E.g 4k frames.
        """
        det = make_detection(x1=0, y1=0, x2=3840, y2=2160)
        result = det.to_cxcysr()
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
    
    def test_fractional_coordinates(self):
        det = make_detection(x1=0.333, y1=0.666, x2=10.999, y2=20.444)
        cx, cy, s, r = det.to_cxcysr()
        recovered = Detection.from_cxcysr(cx=cx, cy=cy, s=s, r=r)
        assert pytest.approx(recovered.x1, abs=ATOL) == det.x1
        assert pytest.approx(recovered.y1, abs=ATOL) == det.y1
        assert pytest.approx(recovered.x2, abs=ATOL) == det.x2
        assert pytest.approx(recovered.y2, abs=ATOL) == det.y2

        