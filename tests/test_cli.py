"""The command line's own contract: one module per verb, and the verbs the docs promise."""

import argparse
import ast
import contextlib
import importlib
import importlib.util
import io
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import pytest

from animalpose.cli import MODULES, build_parser, main
from animalpose.cli.base import add_group, module_name
from animalpose.cli.data import MEMBERS as DATA_MEMBERS
from animalpose.cli.evaluate import MEMBERS as EVAL_MEMBERS
from animalpose.cli.export import MEMBERS as EXPORT_MEMBERS
from animalpose.cli.fixtures import MEMBERS as FIXTURE_MEMBERS
from animalpose.selftrain import LabelOptions

# package -> (verb, dest of the leaf subparser, its members)
GROUPS = {"evaluate": ("eval", "check", EVAL_MEMBERS), "data": ("data", "step", DATA_MEMBERS),
          "export": ("export", "what", EXPORT_MEMBERS), "fixtures": ("fixtures", "part", FIXTURE_MEMBERS)}

VERBS = {"train", "finetune", "predict", "label", "behavior"} | {verb for verb, _, _ in GROUPS.values()}


def test_group_members_follow_the_three_name_contract():
    """A check module defines ``NAME``, ``add_arguments``, ``run``, and a docstring to show in ``--help``."""
    for package, (_, _, members) in GROUPS.items():
        for member in members:
            tree = _tree(f"animalpose.cli.{package}.{member}")
            assert _names(tree) >= {"NAME", "add_arguments", "run"}, f"{package}/{member}"
            assert ast.get_docstring(tree), f"{package}/{member} has no summary line for --help"


def test_top_level_verbs_register_themselves():
    for name in MODULES:
        assert "add_subparser" in _names(_tree(f"animalpose.cli.{name}")), name


def test_parser_offers_exactly_the_documented_verbs():
    """Every verb in the README answers, and nothing else is offered."""
    for verb in VERBS:
        with pytest.raises(SystemExit) as stop:
            build_parser().parse_args([verb, "--help"])
        assert stop.value.code == 0
    assert "invalid choice" in _stderr(["nope"])


def test_group_help_lists_every_member(capsys):
    for package, (verb, _, members) in GROUPS.items():
        with pytest.raises(SystemExit):
            build_parser().parse_args([verb, "--help"])
        listing = " ".join(capsys.readouterr().out.split())
        for member in members:
            assert module_name(f"animalpose.cli.{package}.{member}") in listing, f"{verb} {member}"


def test_the_group_verb_stays_short_while_the_package_stays_descriptive():
    assert "eval" in VERBS and "evaluate" not in VERBS
    for package, (verb, dest, members) in GROUPS.items():
        for member in members:
            module = f"animalpose.cli.{package}.{member}"
            argv = [verb, module_name(module), *_smallest_argv(module)]
            args = build_parser().parse_args(argv)
            assert getattr(args, dest) == argv[1] and callable(args.run)


def test_an_unknown_flag_is_rejected_after_the_lazy_flags_are_registered():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["data", "audit", "--data", ".", "--not-a-flag"])


def test_listing_the_commands_does_not_import_the_scoring_harnesses():
    """The 12 checks are listed from source; only the one you ask for is imported."""
    out = _python("import sys, animalpose.cli as c; c.build_parser(); "
                  "print('animalpose.cli.evaluate.real' in sys.modules, 'sklearn' in sys.modules)")
    assert out == "False False"


def test_label_flags_default_to_the_options_object():
    """One default per threshold: argparse reads them off :class:`LabelOptions` instead of restating them."""
    args = build_parser().parse_args(["label", "--root", "videos"])
    for f in fields(LabelOptions):
        assert getattr(args, f.name) == getattr(LabelOptions(), f.name), f.name


def test_a_missing_checkpoint_is_a_one_line_error(capsys):
    code = main(["label", "--root", "data/raw/videos", "--detector", "runs/nope.pt",
                 "--per-source", "1", "--quiet", "--out", "outputs/_pytest_labels"])
    assert code == 1
    assert capsys.readouterr().err == f"error: no checkpoint at {Path('runs/nope.pt')}\n"


def test_add_group_rejects_a_leaf_flag_named_run(tmp_path):
    """``args.run`` is the command itself; a module may not take that name."""
    package = _stub_package(tmp_path)
    top = argparse.ArgumentParser(prog="t").add_subparsers(dest="top")
    add_group(top, "g", package, ("fake",), help="x", dest="what", metavar="WHAT")
    with pytest.raises(ValueError, match="belongs to the command itself"):
        top._name_parser_map["g"].parse_args(["fake"])


def _stub_package(tmp_path: Path) -> str:
    """A one-command package on ``sys.path``, so ``add_group`` has something to register."""
    (pkg := tmp_path / "fakegroup").mkdir()
    (pkg / "__init__.py").write_text('"""A stand-in group."""')
    (pkg / "fake.py").write_text('"""A stand-in check that steals the name of the command itself."""\n'
                                 "NAME = \"fake\"\n\n"
                                 "def add_arguments(parser):\n"
                                 "    parser.add_argument(\"--run\", default=\"\")\n\n"
                                 "def run(args):\n    pass\n")
    sys.path.insert(0, str(tmp_path))
    return "fakegroup"


def _smallest_argv(module: str) -> list[str]:
    """The least that parses a check: one plausible value per argument it declares as required."""
    parser = argparse.ArgumentParser(prog=module)
    importlib.import_module(module).add_arguments(parser)
    argv: list[str] = []
    for a in parser._actions:
        switches, constant = bool(a.option_strings), a.const is not None
        if not a.required or a.dest == "help" or (switches and constant):  # a --flag is never required
            continue
        value = 0 if a.type in (int, float) else next(iter(a.choices or ["x"]), "x")
        count = 0 if switches and constant else (1 if a.nargs in (None, "?", "+", "*") else int(a.nargs))
        argv += [a.option_strings[0]] if switches else []
        argv += [str(value)] * count
    return argv


def _python(code: str) -> str:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          check=True).stdout.strip()


def _stderr(argv: list[str]) -> str:
    err = io.StringIO()
    with contextlib.suppress(SystemExit), contextlib.redirect_stderr(err):
        build_parser().parse_args(argv)
    return err.getvalue()


def _tree(module: str) -> ast.Module:
    spec = importlib.util.find_spec(module)
    return ast.parse(Path(spec.origin).read_text())


def _names(tree: ast.Module) -> set[str]:
    found: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            found.add(node.name)
        elif isinstance(node, ast.Assign):
            found |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            found.add(node.target.id)
    return found
