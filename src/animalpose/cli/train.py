"""``animalpose train`` / ``animalpose finetune``: fit one network from a YAML config."""

from __future__ import annotations

from animalpose.api import MODELS, finetune, train


def add_subparser(sub) -> None:
    t = sub.add_parser("train", help="train one network from a YAML config")
    t.add_argument("model", choices=MODELS)
    t.add_argument("config", help="e.g. configs/pose.yaml")
    t.add_argument("overrides", nargs="*", help="dotted overrides, e.g. train.steps=200 device=cpu")
    t.set_defaults(run=_run_train)

    f = sub.add_parser("finetune", help="train from an existing checkpoint (config key `init`)")
    f.add_argument("model", choices=MODELS)
    f.add_argument("config")
    f.add_argument("init", metavar="INIT", help="the checkpoint to start from, e.g. runs/pose_real.pt")
    f.add_argument("overrides", nargs="*", help="dotted overrides, on top of `init`")
    f.set_defaults(run=_run_finetune)


def _run_train(args) -> None:
    train(args.model, args.config, args.overrides)


def _run_finetune(args) -> None:
    finetune(args.model, args.config, args.init, args.overrides)
