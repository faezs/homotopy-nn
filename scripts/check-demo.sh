#!/usr/bin/env bash
set -euo pipefail

echo "==> Checking Agda demo modules"

ok() { echo "✅ $1"; }
fail() { echo "❌ $1"; exit 1; }

cmd() {
  local file="$1"
  if agda --library-file=./libraries "$file" >/dev/null; then
    ok "$file"
  else
    echo "-- See AUDIT.md for known postulates and gaps"
    fail "$file"
  fi
}

cmd src/Neural/Network/Conservation.agda
cmd src/Neural/Computational/TransitionSystems.agda
cmd src/Neural/Dynamics/IntegratedInformation.agda

echo "\nAll demo modules checked. For full project:"
echo "  agda --library-file=./libraries src/Everything.agda"

