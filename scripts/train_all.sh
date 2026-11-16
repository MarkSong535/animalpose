#!/usr/bin/env bash
# Train every network on synthetic data with the default configs.
# Extra arguments are passed to every run as overrides (e.g. device=cpu train.steps=500).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
$PY -m animalpose train detector configs/detector.yaml "$@" &
$PY -m animalpose train pose configs/pose.yaml "$@" &
wait
for m in temporal temporal3d behavior behavior3d; do
  $PY -m animalpose train "${m%3d}" "configs/$m.yaml" "$@"
done
