"""``animalpose fixtures``: write the torch reference outputs the Rust tests compare against.

Modules define ``NAME``, ``add_arguments(parser)`` and ``run(args)``, run as ``animalpose fixtures <check>``.
"""

from animalpose.cli.base import add_group

MEMBERS = ("tensor", "models", "pipeline")

HELP = "write the reference fixtures the Rust tests read"


def add_subparser(sub) -> None:
    add_group(sub, "fixtures", "animalpose.cli.fixtures", MEMBERS, help=HELP, dest="part",
              metavar="PART", description=__doc__)
