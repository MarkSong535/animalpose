#!/usr/bin/env bash
# Run the suite with coverage and write coverage.xml for codecov. Extra arguments go to pytest.
# Scope and exclusions live in [tool.coverage.*] in pyproject.toml.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
rm -f .coverage coverage.xml
$PY -m pytest -q --cov --cov-report=term "$@"
$PY -m coverage xml -o coverage.xml
echo
echo "weakest modules (see pyproject for what is out of scope):"
$PY -m coverage report | awk 'NR>2 && NF==6 {print $4, $0}' | sed 's/%//' | sort -n | head -12
