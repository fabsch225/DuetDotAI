#!/bin/sh
set -eu
DUET_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$DUET_DIR"
DUET_PYTHON=${DUET_PYTHON:-"$DUET_DIR/.venv/bin/python"}
if [ ! -x "$DUET_PYTHON" ]; then
  echo 'Local environment missing. Follow SETUP.md to create .venv.' >&2
  exit 1
fi
export HF_HOME=${HF_HOME:-"$DUET_DIR/.model-cache"}
export PYTHONDONTWRITEBYTECODE=1
export PYGAME_HIDE_SUPPORT_PROMPT=1
case "${1:-}" in
  --test)
    shift
    exec "$DUET_PYTHON" -m unittest discover -s tests -v "$@"
    ;;
  --demo)
    shift
    exec "$DUET_PYTHON" -u live_duet.py --solo --notes 24 --bpm 80 --lookahead-beats 4 --commit-beats 2 "$@"
    ;;
  *) exec "$DUET_PYTHON" -u live_midi.py "$@" ;;
esac
