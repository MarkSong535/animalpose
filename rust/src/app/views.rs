//! Multiview: which videos are views of one arena, and which camera of the arena's camera model
//! each one is.
//!
//! A recording names each clip after the view its camera records as (`View-1.mp4`, `Top.mp4`, …),
//! so the clips of one stage share a folder. Videos are matched to the model's cameras by those
//! names; numbered names (`Camera-1`, `View-2`, `camera-02`) that are not camera names follow the
//! model's camera order, lowest number first.

use std::path::{Path, PathBuf};

use anyhow::{bail, Result};

use super::recording::safe_name;

/// The label of camera `index` (from 0) before it is named: `Camera 1`, `Camera 2`, …
pub(crate) fn default_view(index: usize) -> String {
    format!("Camera {}", index + 1)
}

/// Videos grouped by folder, in the order each folder first appears.
pub(crate) fn by_folder(videos: &[PathBuf]) -> Vec<Vec<PathBuf>> {
    let mut groups: Vec<Vec<PathBuf>> = Vec::new();
    for video in videos {
        match groups
            .iter_mut()
            .find(|group| group[0].parent() == video.parent())
        {
            Some(group) => group.push(video.clone()),
            None => groups.push(vec![video.clone()]),
        }
    }
    groups
}

fn key(name: &str) -> String {
    safe_name(name).to_lowercase()
}

/// `View-2`, `view 2`, `camera-02` → 2.
fn number(stem: &str) -> Option<usize> {
    let key = key(stem);
    let digits = ["view", "camera", "cam"]
        .iter()
        .find_map(|prefix| key.strip_prefix(prefix))?
        .trim_start_matches(['-', '_']);
    (!digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit()))
        .then(|| digits.parse().ok())
        .flatten()
}

fn stem(video: &Path) -> String {
    video
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Orders `views` as the cameras named `cameras`: by name, or else by view number.
pub(crate) fn match_cameras(views: &[PathBuf], cameras: &[String]) -> Result<Vec<PathBuf>> {
    let found = || views.iter().map(|v| stem(v)).collect::<Vec<_>>().join(", ");
    if views.len() != cameras.len() {
        bail!(
            "{} views for a camera model of {} cameras ({}): {}",
            views.len(),
            cameras.len(),
            cameras.join(", "),
            found()
        );
    }
    let by_name: Option<Vec<PathBuf>> = cameras
        .iter()
        .map(|camera| {
            let mut matches = views.iter().filter(|v| key(&stem(v)) == key(camera));
            let only = matches.next()?;
            matches.next().is_none().then(|| only.clone())
        })
        .collect();
    if let Some(ordered) = by_name {
        return Ok(ordered);
    }
    let mut numbered: Vec<(usize, &PathBuf)> = Vec::new();
    for view in views {
        match number(&stem(view)) {
            Some(n) if !numbered.iter().any(|(m, _)| *m == n) => numbered.push((n, view)),
            _ => bail!(
                "Name the views after the camera model's cameras ({}): {}",
                cameras.join(", "),
                found()
            ),
        }
    }
    numbered.sort_by_key(|(n, _)| *n);
    Ok(numbered.into_iter().map(|(_, v)| v.clone()).collect())
}

/// Two views that would be saved under the same file name.
pub(crate) fn duplicate(views: &[String]) -> Option<&str> {
    views
        .iter()
        .enumerate()
        .find(|(i, view)| views[..*i].iter().any(|other| key(other) == key(view)))
        .map(|(_, view)| view.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paths(names: &[&str]) -> Vec<PathBuf> {
        names.iter().map(PathBuf::from).collect()
    }

    #[test]
    fn folders_group_their_views_in_order() {
        let groups = by_folder(&paths(&["a/View-1.mp4", "b/x.mp4", "a/View-2.mp4"]));
        assert_eq!(
            groups,
            [
                paths(&["a/View-1.mp4", "a/View-2.mp4"]),
                paths(&["b/x.mp4"])
            ]
        );
    }

    #[test]
    fn views_match_cameras_by_name_then_by_number() {
        let cams = ["Top".to_owned(), "Side View".to_owned()];
        let named = paths(&["s/Side-View.mp4", "s/top.mp4"]);
        assert_eq!(
            match_cameras(&named, &cams).unwrap(),
            paths(&["s/top.mp4", "s/Side-View.mp4"])
        );
        let numbered = paths(&["s/View-2.mp4", "s/View-1.mp4"]);
        assert_eq!(
            match_cameras(&numbered, &cams).unwrap(),
            paths(&["s/View-1.mp4", "s/View-2.mp4"])
        );
        let legacy = paths(&["s/camera-02.mp4", "s/camera-01.mp4"]);
        assert_eq!(
            match_cameras(&legacy, &cams).unwrap()[0],
            PathBuf::from("s/camera-01.mp4")
        );
        let unknown = paths(&["s/left.mp4", "s/right.mp4"]);
        assert!(match_cameras(&unknown, &cams)
            .unwrap_err()
            .to_string()
            .contains("Top, Side View"));
        assert!(match_cameras(&paths(&["s/top.mp4"]), &cams).is_err());
        // Cameras 2 and 3 of a rig recorded: still in camera order.
        assert_eq!(
            match_cameras(&paths(&["s/Camera-3.mp4", "s/Camera-2.mp4"]), &cams).unwrap(),
            paths(&["s/Camera-2.mp4", "s/Camera-3.mp4"])
        );
        assert!(match_cameras(&paths(&["s/View-1.mp4", "s/camera-1.mp4"]), &cams).is_err());
    }

    #[test]
    fn views_that_share_a_file_name_are_found() {
        let views = ["View 1".to_owned(), "Top".to_owned(), "view-1".to_owned()];
        assert_eq!(duplicate(&views), Some("view-1"));
        assert_eq!(duplicate(&views[..2]), None);
    }
}
