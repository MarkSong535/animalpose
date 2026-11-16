"""The shape of a command, and the two helpers that register them.

A command module is one file with three things: ``NAME`` (its verb), ``add_arguments(parser)`` and
``run(args)``; its one-line docstring becomes the ``--help`` summary. A group (``eval``, ``data``,
``export``, ``fixtures``) is a package whose ``MEMBERS`` lists those modules; :func:`add_group` builds
the nested parser from them.

The group registry reads ``NAME`` and the summary out of the source with :mod:`ast` rather than
importing the module, and imports it only when its verb is selected — so ``animalpose --help`` lists
every check without loading torch.

:func:`add_optionals` builds flags from a dataclass, so a threshold has one default and one place to
document it; see :class:`animalpose.selftrain.LabelOptions`.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
from collections.abc import Callable, Sequence
from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any, Protocol


class Command(Protocol):
    """What a command module must provide (checked when the verb runs, not at import)."""

    NAME: str

    def add_arguments(self, parser: Any) -> None: ...

    def run(self, args: Any) -> None: ...


MISSING_HELP = "see the module docstring"


def module_summary(module: str) -> str:
    """The first line of a module's docstring, read from its source without importing it."""
    return (ast.get_docstring(_source_tree(module)) or "").strip().split("\n")[0] or MISSING_HELP


def module_name(module: str) -> str:
    """A module's ``NAME`` literal, read from its source without importing it."""
    for node in _source_tree(module).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "NAME" for t in node.targets):
            value = ast.literal_eval(node.value)
            if isinstance(value, str):
                return value
    raise ValueError(f"{module} defines no NAME string")


def _source_tree(module: str) -> ast.Module:
    spec = importlib.util.find_spec(module)
    if spec is None or not spec.origin:
        raise ImportError(f"command module {module} not found")
    return ast.parse(Path(spec.origin).read_text(), filename=spec.origin)


def add_group(sub: Any, name: str, package: str, members: Sequence[str], *, help: str, dest: str,
              metavar: str, description: str | None = None) -> None:
    """A ``animalpose <name> <member>`` parser tree, one module per member.

    ``members`` are module names in ``package``; the verb each one answers to is its ``NAME``, and the
    line shown here is its docstring — both read from source, so listing the group stays cheap.
    """
    group = sub.add_parser(name, help=help, description=description,
                           formatter_class=argparse.RawDescriptionHelpFormatter)
    leaves = group.add_subparsers(dest=dest, required=True, metavar=metavar, parser_class=LazyParser)
    for member in members:
        module = f"{package}.{member}"
        leaf = leaves.add_parser(module_name(module), help=module_summary(module))

        def setup(module=module, leaf=leaf) -> None:
            importlib.import_module(module).add_arguments(leaf)
            if any(a.dest == "run" for a in leaf._actions):
                raise ValueError(f"{module} adds a --run flag: that name belongs to the command itself")

        leaf.setup = setup
        leaf.set_defaults(run=_lazy_run(module))


class LazyParser(argparse.ArgumentParser):
    """A subcommand that registers its flags only when it is the verb that was chosen (or asked about)."""

    setup: Callable[[], None] | None = None

    def parse_known_args(self, args: Sequence[str] | None = None,
                         namespace: argparse.Namespace | None = None) -> tuple[argparse.Namespace, list[str]]:
        if self.setup is not None:
            self.setup()
            self.setup = None
        return super().parse_known_args(args, namespace)


def _lazy_run(module: str):
    """Defer importing a command module until its verb is the one that was asked for."""

    def run(args: Any) -> None:
        mod = importlib.import_module(module)
        entry = getattr(mod, "run", None)
        if not callable(entry):
            raise RuntimeError(f"{module} defines no run(args)")
        entry(args)

    return run


def add_optionals(parser: Any, options: type, *, skip: Sequence[str] = ()) -> None:
    """One flag per dataclass field: name from the field, default from the field, help from its metadata.

    Field metadata keys: ``help`` (the flag help), ``flag`` (a name other than ``--field-name``),
    ``nargs``, ``action`` (for the ``store_true``/``store_false`` cases), ``metavar``.
    """
    for f in fields(options):
        if f.name in skip:
            continue
        meta = f.metadata
        flag = meta.get("flag", f"--{f.name.replace('_', '-')}")
        kwargs: dict[str, Any] = {"dest": f.name, "default": f.default,
                                  "help": meta.get("help", MISSING_HELP)}
        if (action := meta.get("action")):
            kwargs["action"] = action
        elif (kind := meta.get("type") or _bare_type(f)) is not None:
            kwargs["type"] = kind
        if (choices := meta.get("choices")) is not None:
            kwargs["choices"] = choices
        if (nargs := meta.get("nargs")) is not None:
            kwargs["nargs"] = nargs
        if (metavar := meta.get("metavar")) is not None:
            kwargs["metavar"] = metavar
        if f.default is MISSING:
            kwargs["required"] = True
            kwargs.pop("default")
        parser.add_argument(flag, **kwargs)


def build(options: type, args: Any, *, skip: Sequence[str] = ()) -> Any:
    """A ``options`` instance from the parsed namespace, taking only the fields it declares."""
    return options(**{f.name: getattr(args, f.name) for f in fields(options) if f.name not in skip})


def _bare_type(f) -> type | None:
    """The scalar type of a plain ``int``/``float``/``str`` annotation (a name after
    ``from __future__ import annotations``, so the name is looked up too). A union such as
    ``str | Path`` gets no type: the flag keeps the string and the library decides what it means."""
    if isinstance(f.type, type):
        return f.type if f.type in (int, float, str) else None
    if isinstance(f.type, str):
        return {"int": int, "float": float, "str": str}.get(f.type.strip())
    return None
