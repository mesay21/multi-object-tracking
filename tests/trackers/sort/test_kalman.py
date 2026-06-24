"""
Unit tests for trackers.sort.kalman.KalmanBoxTracker

Tests covered:
    - State seeded correctly
    - predict() advances state and returns detection
    - update() corrects state towards measurement
    - combined predict/update cycles
    - P stays symmetric positive semi-definite
    - get_state() reads state without advancing
    - prametric dt scales velocity correctly
    - custom noise params affect K and convergence
    - degenarate inputs don't crash
    - state vector and covariance properties
"""

from __future__ import annotations
from shutil import make_archive
import traceback
from unittest import result

from _pytest.monkeypatch import K
import numpy as np
import pytest

from core.detection import Detection
from trackers.sort.kalman import KalmanBoxTracker

ATOL = 1e-4 #Absolute tolerance for float comparisions

def make_detection(
    x1: float = 100.0,
    y1: float = 200.0,
    x2: float = 200.0,
    y2: float = 400.0,
    score: float = 0.0,
    class_id: int = 1.0
) -> Detection:
    return Detection(x1=x1, y1=y1, x2=x2, y2=y2, score=score, class_id=class_id)

def make_tracker(**kwargs) -> KalmanBoxTracker:
    return KalmanBoxTracker(make_detection(), **kwargs)

def assert_valid_detections(det: Detection) -> None:
    """
    Assert a Detection has finite values and positive dimensions
    """

    for val in [det.x1, det.y1, det.x2, det.y2]:
        assert np.isfinite(val), f"Non-finite value in detection {det}"
    
    assert det.x2 > det.x1, f"x2 must be > x1: {det}"
    assert det.y2 > det.y1, f"y2 must be > y1: {det}"

def assert_symmetric(matrix: np.ndarray, atol: float = 1e-8) -> None:
    np.testing.assert_allclose(matrix, matrix.T, atol==atol, err_msg="Matrix not symmetric")


def assert_positive_semi_definite(matrix: np.ndarray) -> None:
    eigen_values = np.linalg.eigvalsh(matrix)
    assert np.all(eigen_values >= -1e-8), (
        f"Matrix is not positive semi-definite, MIn eigen value: {eigen_values.min():.6e}"
    )

class TestInitialization:
    def test_state_vector_shape(self):
        tracker = make_tracker()
        assert tracker.state_vector.shape == (7,)
    
    def test_state_seeded_from_detection(self):
        dets = make_detection(x1=100, y1=200, x2=200, y2=400)
        tracker = KalmanBoxTracker(dets)
        cx, cy, s, r = dets.to_cxcysr()
        x = tracker.state_vector

        assert pytest.approx(x[0], abs=ATOL) == cx
        assert pytest.approx(x[1], abs=ATOL) == cy
        assert pytest.approx(x[2], abs=ATOL) == s
        assert pytest.approx(x[3], abs=ATOL) == r
    
    def test_velocities_zero_at_birth(self):
        tracker = make_tracker()
        x = tracker.state_vector
        np.testing.assert_allclose(x[4:], 0.0, atol=ATOL)
    
    def test_covariance_shape(self):
        tracker = make_tracker()
        assert tracker.covariance.shape == (7, 7)
    
    def test_covariance_diag_at_birth(self):
        tracker = make_tracker()
        P = tracker.covariance
        off_diag = P - np.diag(np.diag(P))
        np.testing.assert_allclose(off_diag, 0.0, atol=ATOL)
    
    def test_default_dt(self):
        tracker = make_tracker()
        assert tracker.dt == 1.0
    
    def test_custom_dt(self):
        tracker = make_tracker(dt=0.5)
        assert tracker.dt == 0.5
    
    def test_get_state_matches_initial_detection(self):
        det = make_detection()
        tracker = KalmanBoxTracker(det)
        state = tracker.get_state()
        assert pytest.approx(state.x1, abs=ATOL) == det.x1
        assert pytest.approx(state.y1, abs=ATOL) == det.y1
        assert pytest.approx(state.x2, abs=ATOL) == det.x2
        assert pytest.approx(state.y2, abs=ATOL) == det.y2

class TestPredict:

    def test_returns_detection(self):
        tracker = make_tracker()
        result = tracker.predict()
        assert isinstance(result, Detection)
    
    def test_predicted_detection_is_valid(self):
        tracker = make_tracker()
        result = tracker.predict()
        assert_valid_detections(result)

    def test_zero_velocity_predict_is_stationary(self):
        """
        Prediction should return the same box with zero velocity.
        """
        det = make_detection(x1=100, y1=200, x2=200, y2=400)
        tracker = KalmanBoxTracker(det)
        predicted = tracker.predict()
        assert pytest.approx(predicted.x1, abs=ATOL) == det.x1
        assert pytest.approx(predicted.y1, abs=ATOL) == det.y1
        assert pytest.approx(predicted.x2, abs=ATOL) == det.x2
        assert pytest.approx(predicted.y2, abs=ATOL) == det.y2
    
    def test_predict_advances_state_with_velocity(self):
        """
        Manually inject velocity and confirm prediction moves the box
        """
        tracker = make_tracker()
        #Inject known velocity cx' = 10, cy' = 5
        tracker._x[4, 0] = 10.0
        tracker._x[5, 0] = 5.0
        cx_before = tracker._x[0, 0]
        cy_before = tracker._x[1, 0]
        tracker.predict()
        cx_after = tracker._x[0, 0]
        cy_after = tracker._x[1, 0]

        assert pytest.approx(cx_after, abs=ATOL) == cx_before + 10.0
        assert pytest.approx(cy_after, abs=ATOL) == cy_before + 5.0
    
    def test_predict_inflates_covariance(self):
        """
        P should grow after predict (no measurement to constrain it)
        """
        tracker = make_tracker()
        p_trace_before = np.trace(tracker.covariance)
        tracker.predict()
        p_trace_after = np.trace(tracker.covariance)

        assert p_trace_after > p_trace_before
    
    def test_predict_convariance_remains_symmetric(self):
        tracker = make_tracker()
        for _ in range(10):
            tracker.predict()
        assert_symmetric(tracker.covariance)
    
    def test_covariance_remains_psd(self):
        tracker = make_tracker()
        for _ in range(10):
            tracker.predict()
        assert_positive_semi_definite(tracker.covariance)
    
    def test_multiple_predictions_without_update(self):
        """
        Tracker should not crach over many predict only steps.
        """
        tracker = make_tracker()
        for _ in range(100):
            result = tracker.predict()

        assert_valid_detections(result)
    
    def test_predict_dt_scales_velocity(self):
        """
        With dt=2 position should advance twice
        """
        det = make_detection()
        tracker_dt1 = KalmanBoxTracker(det, dt=1.0)
        tracker_dt2 = KalmanBoxTracker(det, dt=2.0)

        #Inject same velocity

        for t in [tracker_dt1, tracker_dt2]:
            t._x[4, 0] = 10.0 #cx
            t._x[5, 0] = 5.0 #cy
        
        pred_dt1 = tracker_dt1.predict()
        pred_dt2 = tracker_dt2.predict()

        cx_dt1 = (pred_dt1.x1 + pred_dt1.x2) / 2
        cx_dt2 = (pred_dt2.x1 + pred_dt2.x2) / 2

        assert pytest.approx(cx_dt2 - cx_dt1, abs=ATOL) == 10.0 # 2*10 - 1*10

class TestUpdate:

    def test_update_pulls_state_toward_measurement(self):
        """
        After update, state should be closer to measurement than before. 
        """

        tracker = make_tracker()
        #Shift measurement far from current state
        shifted = make_detection(x1=300, y1=400, x2=400, y2=600)
        cx_before = tracker._x[0, 0]
        cx_measurement = (shifted.x1 + shifted.x2) / 2
        tracker.update(shifted)
        cx_after = tracker._x[0, 0]
        assert abs(cx_after - cx_measurement) < abs(cx_before - cx_measurement)
    
    def test_update_reduces_covariance_trace(self):
        """
        P trace should shrink after an update step.
        """
        tracker = make_tracker()
        tracker.predict()
        p_trace_before = np.trace(tracker.covariance)
        tracker.update(make_detection())
        p_trace_after = np.trace(tracker.covariance)
        assert p_trace_after < p_trace_before

    def test_update_covariance_remains_symmetric(self):
        tracker = make_tracker()

        for _ in range(20):
            tracker.predict()
            tracker.update(make_detection())
        assert_symmetric(tracker.covariance)
    
    def test_update_covariance_remains_psd(self):
        tracker = make_tracker()
        for _ in range(20):
            tracker.predict()
            tracker.update(make_detection())
        assert_positive_semi_definite(tracker.covariance)
    
    def test_repeated_same_measurement_converges(self):
        """
        Repeated updates to the same detection should converge state to it.
        """
        det = make_detection(x1=100, y1=200, x2=200, y2=400)
        tracker = KalmanBoxTracker(det)
        target = make_detection(x1=150, y1=250, x2=250, y2=450)
        for _ in range(100):
            tracker.predict()
            tracker.update(target)
        state = tracker.get_state()
        assert pytest.approx(state.x1, abs=ATOL) == target.x1
        assert pytest.approx(state.y1, abs=ATOL) == target.y1
        assert pytest.approx(state.x2, abs=ATOL) == target.x2
        assert pytest.approx(state.y2, abs=ATOL) == target.y2
    
class TestPredictUpdateCycle:
    
    def test_constant_velocity_tracking(self):
        """
        Tracker should follow a linearly moving box after warmup.
        """

        dx = 5.0 # pixels perframe
        #Seed tracker at frame zero
        det_0 = make_detection(x1=100, y1=200, x2=200, y2=400)
        tracker = KalmanBoxTracker(det_0)
        #Feed 30 frames of linearly moving box
        for i in range(1, 31):
            det_i = make_detection(
                x1=det_0.x1 + dx * i,
                y1=det_0.y1,
                x2=det_0.x2 + dx * i,
                y2=det_0.y2
            )
            tracker.predict()
            tracker.update(det_i)
        #After 30 updates the tracker should be close to the true position
        state = tracker.get_state()
        expected_x1 = det_0.x1 + dx * 30
        assert pytest.approx(state.x1, abs=ATOL) == expected_x1
    
    def test_valid_state_across_long_sequence(self):
        tracker = make_tracker()
        for i in range(200):
            det = make_detection(x1=100 + i, y1=200, x2=200 + i, y2=400)
            tracker.predict()
            tracker.update(det)
        assert_valid_detections(tracker.get_state())
    
    def test_predict_only_then_update_recovers(self):
        """
        Test update recovers after several predict steps (i.e missing measurement)
        """
        tracker = make_tracker()
        for _ in range(10):
            tracker.predict()
        
        tracker.update(make_detection())
        assert_valid_detections(tracker.get_state())

class TestCovarianceProperties:
    
    def test_initial_covariance_is_symmetric(self):
        assert_symmetric(make_tracker().covariance)
    
    def test_initial_covariance_is_psd(self):
        assert_positive_semi_definite(make_tracker().covariance)
    
    def test_covariance_stays_psd(self):
        """
        Test covariance stays PSD after predict/update cycles.
        """
        tracker = make_tracker()
        for i in range(50):
            tracker.predict()
            #Simulate missing measurement
            if i%5 != 0:
                det = make_detection(x1=100 + i, y1=200, x2=200 + i, y2=400)
                tracker.update(det)
            assert_positive_semi_definite(tracker.covariance)
    
    def test_covariance_stays_symmetric(self):
        """
        Test covariance stays symmetric after predict/update cycles.
        """
        tracker = make_tracker()
        for i in range(50):
            tracker.predict()
            #Simulate missing measurement
            if i%5 != 0:
                det = make_detection(x1=100 + i, y1=200, x2=200 + i, y2=400)
                tracker.update(det)
            assert_symmetric(tracker.covariance)

class TestGetStates:
    
    def test_get_state_returns_detection(self):
        assert isinstance(make_tracker().get_state(), Detection)
    
    def test_get_state_doesnot_advance_state(self):
        tracker = make_tracker()
        x_before = tracker.state_vector.copy()
        tracker.get_state()
        np.testing.assert_array_equal(tracker.state_vector, x_before)
    
    def test_get_state_doesnot_change_covariance(self):
        tracker = make_tracker()
        p_before = tracker.covariance.copy()
        tracker.get_state()
        np.testing.assert_array_equal(tracker.covariance, p_before)
    
    def test_get_state_consistent_with_predict(self):
        """
        After predict(), get_state should return same box as predict()
        """
        tracker = make_tracker()
        predicted = tracker.predict()
        state = tracker.get_state()
        assert pytest.approx(predicted.x1, abs=ATOL) == state.x1
        assert pytest.approx(predicted.y1, abs=ATOL) == state.y1
        assert pytest.approx(predicted.x2, abs=ATOL) == state.x2
        assert pytest.approx(predicted.y2, abs=ATOL) == state.y2

class TestDtParameter:
    
    def test_largest_dt_moves_prediction_further(self):
        det = make_detection()
        t1 = KalmanBoxTracker(det, dt=1.0)
        t2 = KalmanBoxTracker(det, dt=3.0)
        for t in [t1, t2]:
            t._x[4, 0] = 10.0 # inject cx velocity
        
        p1 = t1.predict()
        p2 = t1.predict()
        cx1 = p1.x1 + p1.x2
        cx2 = p2.x1 + p2.x2
        assert cx2 > cx1
    
    def test_zero_dt_prediction_is_stationary(self):
        det = make_detection()
        tracker = KalmanBoxTracker(det, dt=0.0)
        tracker._x[4, 0] = 100.0 #Large velocity should have no effect
        predicted = tracker.predict()
        state_cx = (predicted.x1 + predicted.x2) / 2
        det_cx = (det.x1 + det.x2) / 2

        assert pytest.approx(state_cx, abs=ATOL) == det_cx

class TestNoiseTuning:
    
    def test_high_r_scale(self):
        """
        With high measurement noise on scale, the Kalman Gain for scale should be lower
        """
        det = make_detection()
        low_r = KalmanBoxTracker(det, r_scale_scale=1.0)
        high_r = KalmanBoxTracker(det, r_scale_scale=1000.0)

        #Compute Kalman gain
        def kalman_gain(tracker):
            P = tracker._P
            H = tracker._H
            R = tracker._R
            S = H @ P @ H.T + R
            K = np.linalg.solve(S.T, H @ P.T).T

            return K
        k_low = kalman_gain(low_r)
        k_high = kalman_gain(high_r)

        assert k_low[2, 2] > k_high[2, 2]
    
    def test_high_p_vel_scale(self):
        """
        High velocity scale increases inital velocity uncertainity. 
        """
        low_pv = KalmanBoxTracker(make_detection(), p_vel_scale=10.0)
        high_pv = KalmanBoxTracker(make_detection(), p_vel_scale=1000.0)
        assert high_pv.covariance[4, 4] > low_pv.covariance[4, 4]

class TestEdgeCases:
    
    def test_tiny_box_crash(self):
        """
        Tiny box sizes should not crash the filter.
        """
        det = make_detection(x1=0.0, y1=0.0, x2=2.0, y2=2.0, score=0.9, class_id=1)
        tracker = KalmanBoxTracker(det)
        result = tracker.predict()
        assert_valid_detections(result)
    
    def test_large_box_crash(self):
        """
        Large box sizes should not crash the filter.
        """
        det = make_detection(x1=0.0, y1=0.0, x2=3840.0, y2=2160.0, score=0.9, class_id=1)
        tracker = KalmanBoxTracker(det)
        result = tracker.predict()
        assert_valid_detections(result)
    
    def test_no_nan_from_missed_frames(self):
        """
        Many missed frames/measurement should not cause NaN.
        """
        tracker = make_tracker()
        for _ in range(500):
            tracker.predict()
        state = tracker.get_state()
        for val in [state.x1, state.y1, state.x2, state.y2]:
            assert np.isfinite(val)
    def test_far_update_crash(self):
        """
        Update using far detection should not crash the filter.
        """
        tracker = make_tracker()
        far_det = make_detection(x1=5000, y1=5000, x2=5100, y2=5200)
        tracker.predict()
        tracker.update(far_det)
        assert_valid_detections(tracker.get_state())
    
    def test_square_detection(self):
        det = make_detection(x1=0.0, y1=0.0, x2=64.0, y2=64.0)
        tracker = KalmanBoxTracker(det)
        result = tracker.predict()
        assert_valid_detections(result)

class TestStateAccessors:
    ...