"""The animalpose command line: one entry point, one module per task.

    animalpose train|finetune|label|predict|behavior|eval|data|export|fixtures

Each module in this package registers its verbs and calls the same library functions that
:mod:`animalpose.api` exposes, so the CLI and :command:`python -c "import animalpose"` do exactly
the same thing.
"""

from __future__ import annotations

import argparse
import importlib
import sys

MODULES = ("train", "predict", "label", "behavior", "evaluate", "data", "export", "fixtures")

DESCRIPTION = """animalpose: pose, tracking and behavior for rodents.

  train / finetune  fit a network from a YAML config (optionally from an existing checkpoint)
  label             pseudo-label boxes and keypoints from unlabeled recordings
  predict           detect, track and classify a video
  behavior          cluster tracks, name clusters, write behavior labels
  eval              score a model, with or without ground truth
  data              describe a camera, prepare and convert datasets
  export            write safetensors for the Rust runtime, TorchScript / ONNX for others
  fixtures          write the torch reference fixtures the Rust tests read
"""


def build_parser() -> argparse.ArgumentParser:
    """The whole command tree, importing one module per verb group."""
    from animalpose import __version__

    ap = argparse.ArgumentParser(prog="animalpose", description=DESCRIPTION,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"animalpose {__version__}")
    sub = ap.add_subparsers(dest="command", required=True, metavar="COMMAND")
    for name in MODULES:
        importlib.import_module(f"animalpose.cli.{name}").add_subparser(sub)
    return ap


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` (default ``sys.argv``) and run the selected command."""
    args = build_parser().parse_args(argv)
    try:
        args.run(args)
    except (OSError, ValueError) as exc:  # a wrong path or config is a mistake, not a crash
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
