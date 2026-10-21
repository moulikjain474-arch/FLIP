# source this before running anything:  source env.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export PYTHONPATH="$ROOT/core:$ROOT/experiments:$ROOT/analysis:$ROOT/claims:$ROOT/tools:$PYTHONPATH"
