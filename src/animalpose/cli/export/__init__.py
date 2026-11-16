"""``animalpose export``: write the weights the Rust runtime and ONNX / TorchScript consumers read.

Modules define ``NAME``, ``add_arguments(parser)`` and ``run(args)``, run as ``animalpose export <check>``.
"""

from animalpose.cli.base import add_group

MEMBERS = ("models", "safetensors", "import_safetensors")

HELP = "export TorchScript / ONNX / safetensors, and import them back"


def add_subparser(sub) -> None:
    add_group(sub, "export", "animalpose.cli.export", MEMBERS, help=HELP, dest="what",
              metavar="WHAT", description=__doc__)
