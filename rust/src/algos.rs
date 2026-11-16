//! Pure-Rust port of the algorithmic parts of the `animalpose` Python package:
//! blob proposals, heat-map decoding, non-maximum suppression, Kalman tracking,
//! Hungarian association, behavior probability post-processing and per-frame
//! kinematics.
//!
//! Every module mirrors a module under `src/animalpose/`: the same functions,
//! the same names in snake_case, the same defaults and the same semantics. No
//! autograd, no neural nets and no OpenCV: images are the plain buffers of
//! [`crate::algos::image`], matrices the small dense [`crate::algos::linalg::Mat`], and the Hungarian
//! assignment of `scipy.optimize.linear_sum_assignment` lives in [`crate::algos::hungarian`].
//!
//! Where the Rust behaviour has to deviate from Python the deviation is marked
//! with a `// differs from Python:` comment at the site of the change, and the
//! main ones are listed in the module documentation below.
//!
//! ```
//! use animalpose::algos::tracker::{Detection, MultiAnimalTracker};
//!
//! let mut tracker = MultiAnimalTracker::with_defaults();
//! let det = Detection::new([0.0, 0.0, 20.0, 20.0], 0.9);
//! let matched = tracker.update(&[det]);
//! assert_eq!(matched.keys().copied().collect::<Vec<_>>(), vec![0]);
//! ```

pub mod background;
pub mod blob;
pub mod decode;
pub mod geometry;
pub mod hungarian;
pub mod image;
pub mod kalman;
pub mod kinematics;
pub mod linalg;
pub mod nms;
pub mod postprocess;
pub mod tracker;

pub use background::{arena_roi, foreground_fraction, median_background, otsu_threshold};
pub use blob::{
    blob_box, connected_components, largest_moving_blob, morph_close, morph_open, BlobOptions,
    BlobProposer, Component,
};
pub use decode::{
    detector_decode, pose_decode, pose_decode_at, pose_decode_pairs, DetectorDecode, HeatMaps,
    PoseDecode,
};
pub use geometry::{
    apply_affine, box_cs_to_xyxy, box_iou, box_iou_pairwise, box_xyxy_to_cs,
    box_xyxy_to_cs_default, crop_affine, invert_affine, keypoints_to_box, unletterbox_boxes, Box4,
};
pub use hungarian::linear_sum_assignment;
pub use image::{letterbox, letterbox_gray, Bgr, GrayF, Image, ToGray};
pub use kalman::BoxKalman;
pub use kinematics::{body_kinematics, Kinematics};
pub use linalg::Mat;
pub use nms::{merge_candidates, nms, peak_mask, suppress_non_peaks};
pub use postprocess::{
    adapt_prior, fit_temperature, median_smooth, moving_median, temperature_scale,
};
pub use tracker::{
    assign, cos_dist, oks, Assignment, Detection, MultiAnimalTracker, Track, TrackState,
    TrackerConfig,
};
