"""``animalpose eval``: the scoring harnesses, one module per check, run as ``animalpose eval <check>``.

Every module here is a thin harness over :mod:`animalpose.eval` and the pipeline; it defines ``NAME``,
``add_arguments(parser)`` and ``run(args)``.
"""

from animalpose.cli.base import add_group

MEMBERS = ("real", "blur", "unlabeled", "mirror_paws", "mirror_videos", "rat7m_3d", "real_tracking", "real_tracks",
           "crim13_identity", "behavior_loao", "behavior_sessions", "benchmark")

HELP = "score a model, with or without ground truth"


def add_subparser(sub) -> None:
    add_group(sub, "eval", "animalpose.cli.evaluate", MEMBERS, help=HELP, dest="check",
              metavar="CHECK", description=__doc__)
