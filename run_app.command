#!/bin/bash
# Launch from Finder or Terminal without changing the user's shell settings.
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "Create .venv and install requirements first. See START_HERE.md."
  exit 1
fi
if [ "$(uname -s)" = "Darwin" ]; then
  if ! .venv/bin/python -c 'import pyexpat' >/dev/null 2>&1; then
    if [ -f /opt/homebrew/opt/expat/lib/libexpat.1.dylib ]; then
      export DYLD_LIBRARY_PATH="/opt/homebrew/opt/expat/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
    fi
    if ! .venv/bin/python -c 'import pyexpat'; then
      echo "Python's XML library could not load. See the macOS troubleshooting section in START_HERE.md."
      exit 1
    fi
  fi
fi
exec .venv/bin/python -m streamlit run Home.py
