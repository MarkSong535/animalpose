//! Constant-velocity Kalman filter on box centre and size (SORT/DeepSORT style).
//!
//! Port of `animalpose.tracking.kalman.BoxKalman`.

use crate::algos::geometry::Box4;
use crate::algos::linalg::Mat;
use anyhow::Result;

/// Default `pos_std` of `BoxKalman.__init__` (`1 / 20`).
pub const DEFAULT_POS_STD: f64 = 1.0 / 20.0;
/// Default `vel_std` of `BoxKalman.__init__` (`1 / 160`).
pub const DEFAULT_VEL_STD: f64 = 1.0 / 160.0;

/// State `[cx, cy, w, h, vcx, vcy, vw, vh]`; noise scales with box size.
#[derive(Clone, Debug)]
pub struct BoxKalman {
    pos_std: f64,
    vel_std: f64,
    x: Mat,
    p: Mat,
}

impl BoxKalman {
    /// A filter initialised from an xyxy box with the Python default noise
    /// scales (`pos_std = 1/20`, `vel_std = 1/160`).
    pub fn new(bbox: Box4) -> Self {
        Self::with_stds(bbox, DEFAULT_POS_STD, DEFAULT_VEL_STD)
    }

    /// Port of `BoxKalman.__init__(box, pos_std, vel_std)`.
    pub fn with_stds(bbox: Box4, pos_std: f64, vel_std: f64) -> Self {
        let z = to_z(bbox);
        let mut state = z.to_vec();
        state.extend_from_slice(&[0.0; 4]);
        let s = z[2].max(z[3]);
        let mut diag = vec![2.0 * pos_std * s; 4];
        diag.extend_from_slice(&[10.0 * vel_std * s; 4]);
        Self {
            pos_std,
            vel_std,
            x: Mat::column(&state),
            p: Mat::diag(&diag),
        }
    }

    /// Observation of an xyxy box in state space (port of `BoxKalman._to_z`).
    pub fn observation(bbox: Box4) -> [f64; 4] {
        to_z(bbox)
    }

    /// The filtered box, xyxy (port of the `box` property).
    pub fn bbox(&self) -> Box4 {
        let (cx, cy) = (self.x.get(0, 0), self.x.get(1, 0));
        let (w, h) = (self.x.get(2, 0).max(1.0), self.x.get(3, 0).max(1.0));
        [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]
    }

    /// Centre velocity `[vcx, vcy]` (port of the `velocity` property).
    pub fn velocity(&self) -> [f64; 2] {
        [self.x.get(4, 0), self.x.get(5, 0)]
    }

    /// Full state `[cx, cy, w, h, vcx, vcy, vw, vh]`.
    pub fn state(&self) -> Vec<f64> {
        self.x.to_vec()
    }

    /// Current covariance matrix.
    pub fn covariance(&self) -> &Mat {
        &self.p
    }

    /// Kalman transition matrix `F` (`_F` in Python).
    fn f() -> Mat {
        let mut m = Mat::eye(8);
        for i in 0..4 {
            m.set(i, i + 4, 1.0);
        }
        m
    }

    /// Observation matrix `H` (`_H` in Python).
    fn h() -> Mat {
        let mut m = Mat::new(4, 8);
        for i in 0..4 {
            m.set(i, i, 1.0);
        }
        m
    }

    /// Advances the state one frame and returns the predicted box
    /// (port of `BoxKalman.predict`).
    pub fn predict(&mut self) -> Box4 {
        let s = self.x.get(2, 0).max(self.x.get(3, 0));
        let mut q = vec![self.pos_std * s; 4];
        q.extend_from_slice(&[self.vel_std * s; 4]);
        let q = Mat::diag(&q);
        let f = Self::f();
        let new_x = &f * &self.x;
        let new_p = &(&(&f * &self.p) * &f.transpose()) + &q;
        self.x = new_x;
        self.p = new_p;
        self.bbox()
    }

    /// Observation residual and its covariance
    /// (port of `BoxKalman.innovation`).
    pub fn innovation(&self, bbox: Box4) -> (Vec<f64>, Mat) {
        let s = self.x.get(2, 0).max(self.x.get(3, 0));
        let r = Mat::diag(&[self.pos_std * s; 4]);
        let h = Self::h();
        let ht = h.transpose();
        let s = &(&(&h * &self.p) * &ht) + &r;
        let hx = &h * &self.x;
        let y: Vec<f64> = to_z(bbox)
            .iter()
            .enumerate()
            .map(|(i, v)| v - hx.get(i, 0))
            .collect();
        (y, s)
    }

    /// Squared Mahalanobis distance of the box centre
    /// (port of `BoxKalman.mahalanobis`).
    ///
    /// // differs from Python: a singular innovation covariance raises in
    /// // `np.linalg.solve`; here it counts as infinitely far, which is what
    /// // the association gate wants anyway.
    pub fn mahalanobis(&self, bbox: Box4) -> f64 {
        let (y, s) = self.innovation(bbox);
        let s2 = s.block(0, 0, 2, 2);
        match s2.solve(&y[..2]) {
            Ok(v) => y[0] * v[0] + y[1] * v[1],
            Err(_) => f64::INFINITY,
        }
    }

    /// Fuses an observation (port of `BoxKalman.update`).
    ///
    /// // differs from Python: a singular innovation covariance is skipped
    /// // instead of raising `LinAlgError`.
    pub fn update(&mut self, bbox: Box4) -> Result<()> {
        let (y, s) = self.innovation(bbox);
        let y = Mat::column(&y);
        let inv_s = s.inverse()?;
        let h = Self::h();
        let ht = h.transpose();
        let p_ht = &self.p * &ht;
        let kg = &p_ht * &inv_s;
        let kg_y = &kg * &y;
        let new_x = &self.x + &kg_y;
        let kg_h = &kg * &h;
        let i8 = Mat::eye(8);
        let new_p = &(&i8 - &kg_h) * &self.p;
        self.x = new_x;
        self.p = new_p;
        Ok(())
    }
}

/// Port of `BoxKalman._to_z`.
fn to_z(bbox: Box4) -> [f64; 4] {
    [
        (bbox[0] + bbox[2]) / 2.0,
        (bbox[1] + bbox[3]) / 2.0,
        (bbox[2] - bbox[0]).max(1.0),
        (bbox[3] - bbox[1]).max(1.0),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, atol: f64) -> bool {
        (a - b).abs() <= atol
    }

    #[test]
    fn kalman_tracks_constant_velocity() {
        // Port of tests/test_tracking.py::test_kalman_tracks_constant_velocity.
        let mut kf = BoxKalman::new([0.0, 0.0, 10.0, 10.0]);
        for t in 1..15i64 {
            kf.predict();
            kf.update([
                2.0 * t as f64,
                t as f64,
                2.0 * t as f64 + 10.0,
                t as f64 + 10.0,
            ])
            .unwrap();
        }
        let pred = kf.predict();
        let expect = [30.0, 15.0, 40.0, 25.0];
        for (a, e) in pred.iter().zip(expect) {
            assert!(close(*a, e, 0.5), "{pred:?} vs {expect:?}");
        }
    }

    #[test]
    fn state_and_noise_scale_with_the_box() {
        let kf = BoxKalman::new([0.0, 0.0, 20.0, 10.0]);
        let state = kf.state();
        assert_eq!(&state[..4], &[10.0, 5.0, 20.0, 10.0]);
        assert_eq!(&state[4..], &[0.0, 0.0, 0.0, 0.0]);
        // P = diag([2*pos_std*s]*4 + [10*vel_std*s]*4) with s = max(w, h) = 20
        let expected = [2.0 * DEFAULT_POS_STD * 20.0, 10.0 * DEFAULT_VEL_STD * 20.0];
        assert!(close(kf.covariance().get(0, 0), expected[0], 1e-12));
        assert!(close(kf.covariance().get(4, 4), expected[1], 1e-12));
        assert!(close(kf.covariance().get(0, 4), 0.0, 1e-12));
    }

    #[test]
    fn degenerate_boxes_keep_a_unit_size() {
        let kf = BoxKalman::new([5.0, 5.0, 5.0, 5.0]);
        let b = kf.bbox();
        assert!(close(b[2] - b[0], 1.0, 1e-12));
        assert!(close(
            BoxKalman::observation([0.0, 0.0, -3.0, -4.0])[2],
            1.0,
            1e-12
        ));
    }

    #[test]
    fn innovation_and_gate_react_to_the_prediction() {
        let mut kf = BoxKalman::new([0.0, 0.0, 10.0, 10.0]);
        for t in 1..15i64 {
            kf.predict();
            kf.update([
                2.0 * t as f64,
                t as f64,
                2.0 * t as f64 + 10.0,
                t as f64 + 10.0,
            ])
            .unwrap();
        }
        let v = kf.velocity();
        assert!(close(v[0], 2.0, 0.1) && close(v[1], 1.0, 0.1), "{v:?}");
        let pred = kf.bbox();
        let on = kf.mahalanobis([pred[0] + 1.0, pred[1], pred[2] + 1.0, pred[3]]);
        let off = kf.mahalanobis([pred[0] + 200.0, pred[1], pred[2] + 200.0, pred[3]]);
        assert!(on < 50.0, "{on}");
        assert!(off > 50.0, "{off}");
        let (y, s) = kf.innovation(pred);
        assert!(y.iter().all(|v| v.abs() < 1e-6));
        assert!(s.get(0, 0) > 0.0);
    }
}
