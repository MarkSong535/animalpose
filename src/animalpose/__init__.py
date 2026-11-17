"""animalpose: species-agnostic multi-animal pose estimation.

The calls most people need, importable from the top of the package::

    import animalpose

    animalpose.train("pose", "configs/pose.yaml")
    animalpose.finetune("pose", "configs/pose.yaml", init="weights/pose_general.pt")
    animalpose.label("data/raw/videos", include=["myvideo"], out="data/processed/myvideo_pseudo")
    animalpose.predict("myvideo.mp4", config="configs/pipeline.yaml")
    animalpose.cluster_tracks(tracks="outputs/myvideo/tracks.npz", video="myvideo.mp4", out="outputs/myvideo/clusters")

``animalpose.behavior`` is the classifier subpackage, not a function: bootstrap labels with
``cluster_tracks`` / ``write_naming`` / ``label_clusters``, the three steps of
``animalpose behavior cluster|name|label``.

Subpackages
-----------
skeleton   canonical keypoint vocabulary and per-species / per-dataset mappings
detector   anchor-free animal detector (stage 1)
pose       top-down keypoint network over the canonical vocabulary (stage 2)
temporal   LSTM track smoother / gap filler with imputation flags
tracking   Hungarian multi-animal tracker with re-identification
behavior   frame-wise behavior classifier, and bootstrapping labels from tracks
selftrain  pseudo-labeling of unlabeled recordings
multiview  cameras, calibration, triangulation, cross-view identity fusion
eval       scoring harnesses, with and without ground truth
export     TorchScript / ONNX export and parity checks
cli        the ``animalpose`` command line, a thin layer over this API
"""

from animalpose.device import select_device
from animalpose.skeleton import CANONICAL, Skeleton

__all__ = ["CANONICAL", "Skeleton", "cluster_tracks", "finetune", "label", "label_clusters", "predict",
           "select_device", "train", "write_naming"]
__version__ = "1.0.0"


def __getattr__(name: str):
    """The verbs come from :mod:`animalpose.api`, imported on first use (it pulls in torch)."""
    if name in ("train", "finetune", "label", "predict", "cluster_tracks", "write_naming", "label_clusters"):
        from animalpose import api

        return getattr(api, name)
    raise AttributeError(f"module 'animalpose' has no attribute {name!r}")
