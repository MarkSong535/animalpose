"""``animalpose data``: describe a camera and turn raw datasets into training data.

Modules define ``NAME``, ``add_arguments(parser)`` and ``run(args)``, run as ``animalpose data <check>``.
"""

from animalpose.cli.base import add_group

MEMBERS = ("audit", "calibrate", "prepare", "convert", "download")

HELP = "measure a camera profile and prepare datasets"


def add_subparser(sub) -> None:
    add_group(sub, "data", "animalpose.cli.data", MEMBERS, help=HELP, dest="step",
              metavar="STEP", description=__doc__)
