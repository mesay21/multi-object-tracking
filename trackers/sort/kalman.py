"""
Implements a constant velocity linear Kalman filter for bounding box tracking
using SORT algorithm.

    x = [cx, cy, s, r, cx', cy', s'] 
Note: SORT assumes aspect ratio is constant.
where:
    cx/cy: bounding box center pixel coordinates
    s: bounding box area
    r: bounding box aspect ratio
    cx'/cy'/s': velocities of center and area

The measurement is the bounding box center pixel coordinates, area, and aspect ratio.
    z = [cx, cy, s, r]
Note: lifecycle and metadata related to the tracker are not handeled here.
"""

from __future__ import annotations

import numpy as np

from core.detection import Detection

class KalmanBoxTracker:
    """
    Constant-velocity Kalman filter for a single bounding box.
    State vector x: [cx, cy, s, r, cx', cy', s']
    Measurement vector z: [cx, cy, s, r]

    Args:
        detection: Initial detection used to initialize the tracker.
        dt: Time step between frames. Default 1. (One frame)
        p_pos_scale: Initial covariance scale for position/scale/ratio. Default 10.0 
        p_vel_scale: Inital covariance scale for position/scale velocity. Default 1000.0 
        q_pos_scale: Process noise scale for position/scale/ratio. Default 1.0
        q_vel_scale: Process noise scale for position velocity. Default 0.01
        q_scale_vel_scale: Process noise for scale velocity. Default 0.0001
        r_pos_scale: Measurement noise scale for position. Default 1.0
        r_scale_scale: Measurement noise scale for scale. Default 10.0 (higher than 
        poisition noise because detectors are less precise on box size)
    Note: Default values are taken from the SORT paper
    Usage example:
        tracker = KalmanBoxTracker(detection)
        prediction = tracker.predict() #Prediction step
        tracker.update(new_detection) #Update prediction with new measurement
        state = tracker.get_state() #Read state without advancing


    """
    def __init__(
        self, 
        detection: Detection,
        dt: float = 1,
        p_pos_scale: float = 10.0,
        p_vel_scale: float = 1000.0, #High uncertainity because we don't observe velocity at birth
        q_pos_scale: float = 1.0,
        q_vel_scale: float = 0.01,
        q_scale_vel_scale: float = 0.0001,
        r_pos_scale: float = 1.0,
        r_scale_scale: float = 10.0
        ) -> None:

        self.detection = detection
        self.dt = dt
        self.p_pos_scale = p_pos_scale
        self.p_vel_scale = p_vel_scale
        self.q_pos_scale = q_pos_scale
        self.q_vel_scale = q_vel_scale
        self.q_scale_vel_scale = q_scale_vel_scale
        self.r_pos_scale = r_pos_scale
        self.r_scale_scale = r_scale_scale
        self._dim_x = 7
        self._dim_z = 4
        #Initalize filter parameters
        self._x = self._init_state()
        self._F = self._build_state_transition_matrix()
        self._H = self._build_observation_matrix()
        self._P = self._build_state_covariance()
        self._Q = self._build_process_noise_covariance()
        self._R = self._build_measurement_noise_covariance()
        #We use Joseph form for numarically stable covariance update 
        self._I = np.eye(self._dim_x, dtype=np.float64)


    def _init_state(self) -> np.ndarray:
        """
        Initialize state vector using the initial detection.
        """
        cx, cy, s, r = self.detection.to_cxcysr()
        x = np.array(
            [cx, cy, s, r, 0.0, 0.0, 0.0], dtype=np.float64
            ).reshape(-1, 1) #Shape (7, 1)
        
        return x
    
    def _build_state_transition_matrix(self) -> np.ndarray:
        """
        Build state transition (F) matrix (7 x 7).
        Constant velocity model: position += self.dt * velocity
        Aspect ratio (r) has no velocity term
        """
        F = np.eye(self._dim_x, dtype=np.float64)
        F[0, 4] = self.dt # cx += cx' * dt
        F[1, 5] = self.dt # cy += cy' * dt
        F[2, 6] = self.dt # s += s' * dt

        return F

    
    def _build_observation_matrix(self) -> np.ndarray:
        """
        Build observation (H) matrix (4, 7). Maps states to measurement space.
        """
        H = np.zeros(shape=(self._dim_z, self._dim_x), dtype=np.float64)
        H[:4, :4] = np.eye(self._dim_z)

        return H
    
    def _build_state_covariance(self) -> np.ndarray:
        """
        Build initial covariance matrix P (7 x 7)
        """
        p_diag = np.array(
            [
                self.p_pos_scale, #cx
                self.p_pos_scale, #cy
                self.p_pos_scale, #s
                self.p_pos_scale, #r
                self.p_vel_scale, #cx'
                self.p_vel_scale, #cy'
                self.p_vel_scale, #s'
            ],
            dtype=np.float64
        )
        P = np.diag(p_diag)

        return P
    
    def _build_process_noise_covariance(self) -> np.ndarray:
        """
        Build process noise covariance matrix - Q (7 x 7)
        """
        q_diag = np.array(
            [
                self.q_pos_scale, #cx
                self.q_pos_scale, #cy
                self.q_pos_scale, #s
                self.q_pos_scale, #r
                self.q_vel_scale, #cx'
                self.q_vel_scale, #cy'
                self.q_scale_vel_scale, #s'
            ],
            dtype=np.float64
        )
        Q = np.diag(q_diag)

        return Q
    
    def _build_measurement_noise_covariance(self) -> np.ndarray:
        """
        Build measurement noise convariance matrix R (4 x 4)
        """

        r_diag = np.array(
            [
                self.r_pos_scale, #cx
                self.r_pos_scale, #cy
                self.r_scale_scale, #s
                self.r_scale_scale, #r
            ],
            dtype=np.float64
        )
        R = np.diag(r_diag)

        return R
    
    def predict(self) -> Detection:
        """
        Adavnce the filter one time step using motion model. 
        Applies the predict step.
            x = F @ x
            P = F @ P @ F.T + Q
        Returns:
            Predicted Detection in [x1, y1, x2, y2] space. score and class ID are
            set to 0.0/0 since this is a predicted state and not detector output.
        """
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q

        return self._state_to_detection()
    
    def update(self, detection: Detection) -> None:
        """
        Correct filter state predictions using a new measurement.
        Applies the standard Kalman Filter update equations.
            y = z - H @ x (innovation)
            S = H @ P @ H.T + R (innovation covariance)
            K = P @ H.T @ S^-1 (kalman gain)
            x = x + K@Y
            P = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T (Joseph form)
        The Joseph form is numerically stable (instead of the simple (I - K @ H) @ P) 
        and keeps P symmetric positive semi-definite even with floating point rounding 

        Args:
            detection: Matched detections providing the measurement z.
        """
        z = detection.to_cxcysr().reshape(-1, 1).astype(np.float64)

        #Innovation
        y = z - self._H @ self._x
        #innovation covariance
        S = self._H @ self._P @ self._H.T  + self._R

        #Kalman gain
        #Use numpy linalgera solve for stability instead of inverting S
        # S.T @ K.T = H @ P.T
        K = np.linalg.solve(S.T, self._H @ self._P.T).T

        #Update state
        self._x = self._x + K @ y

        #Update convariance 
        I_KH = self._I - K @ self._H
        self._P = I_KH @ self._P @ I_KH.T + K @ self._R @ K.T
    
    def get_state(self) -> Detection:
        """
        Return current state as a Detection.
        Only used to read current state between predict/update calls.
        Retruns:
            Detection in [x1, y1, x2, y2] space derived from current x.
        """
        return self._state_to_detection()
    
    def _state_to_detection(self) -> Detection:
        """
        Convert internal state vector into a Detection object.
        Reads [cx, cy, s, r] from x and delegates to Detection.from_cxcysr.
        """
        cx, cy, s, r = self._x[:4, 0]
        #Guard against degenrate state
        s = max(s, 1e-6)
        r = max(r, 1e-6)

        return Detection.from_cxcysr(
            cx=float(cx),
            cy=float(cy),
            s=float(s),
            r=float(r),
            score=0.0,
            class_id=0
        )
    
    @property
    def state_vector(self) -> np.ndarray:
        """
        Return a copy of the raw state vector.
        """
        return self._x.flatten().copy()
    
    @property
    def covariance(self) -> np.ndarray:
        """
        Return a copy of the convariance matrix P.
        """
        return self._P.copy()
    
    def __repr__(self) -> str:
        cx, cy, s, r = self._x[:4, 0]
        return (
            f"KalmanBoxTracker("
            f"cx={cx:.1f}, cy={cy:.1f}, "
            f"s={s:.1f}, r={r:.3f}, "
            f"dt={self.dt}"
        )

