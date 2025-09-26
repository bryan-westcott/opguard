#!/bin/bash

# Prep dev environment
# 1. check if sourcing
# 2. create project venv (if not exist)
# 3. activate venv (and check that VIRTUAL_ENV var is set)
# 4. sync dev/notebook dependencies
# 5. install current project as an editable package
# 6. register environment as jupyter kernel

# This script must be sourced, so these checks are invalid
# shellcheck disable=SC2317,SC1091

# Ensure script is sourced, not executed
(return 0 2>/dev/null) || {
  echo "❌ This script must be sourced, not executed."
  echo "   Run it like this:  source dev-init.sh"
  exit 1
}

# Note: this has to be a subshell better trap error in the dev script itself
dev_init() {

  # Save current options
  local old_opts
  old_opts=$(set +o)
  # Always restore options when this function exits
  trap 'eval "$old_opts"; trap - ERR; trap - EXIT' EXIT

  # Set temp options
  # -u (nounset): treat unset variables as an error
  #   Example: echo "$FOO" when FOO is not set → causes immediate failure
  #   Helps catch typos and missing env vars early.
  set -u
  # -o pipefail: makes a pipeline fail if *any* command in it fails
  #   Default in Bash: only the *last* command in a pipeline matters.
  #   Example:
  #     false | true
  #   Without pipefail → exit code 0 (because 'true' succeeded).
  #   With pipefail   → exit code 1 (because 'false' failed).
  #
  #   This is useful when chaining commands with | where every stage matters.
  set -o pipefail

  # trap errors (assume sourced) to indicate dev-init problems
  # Note: trap persist until function succeeds and runs bottom "trap - ERR"
  trap 'echo "❌ Error in dev-init.sh at line $LINENO"; return 1' ERR

  # Resolve directory of this script, then its parent (the project root)
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
  cd "$PROJECT_ROOT" || {
    echo "Error: could not cd to $PROJECT_ROOT" >&2
    return 1 2>/dev/null || true
  }

  # Check for pyproject.toml
  PYPROJECT_FILE="${PROJECT_ROOT?}/pyproject.toml"
  if [ ! -f "$PYPROJECT_FILE" ]; then
    echo "❌ No pyproject.toml found in $PROJECT_ROOT, expected a minimal skeleton" >&2
    { return 1 2>/dev/null; } || exit 1
  fi

  # extract project name (first match of `name = "..."`)
  PROJECT_NAME="$(grep -m1 '^[[:space:]]*name[[:space:]]*=' "$PYPROJECT_FILE" \
      | sed -E 's/.*name[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/')"

  # check if notebook section in pyproject.toml
  if grep -q '^[[:space:]]*notebook[[:space:]]*=' "$PYPROJECT_FILE"; then
    HAS_NOTEBOOK=1
  else
    echo "ℹ️  No 'notebook' group found in $PYPROJECT_FILE; skipping Jupyter setup."
    HAS_NOTEBOOK=0
  fi

  echo "📦 Creating virtual environment with uv..."
  uv venv

  echo "🔗 Activating virtual environment..."
  if ! source .venv/bin/activate; then
    echo "❌ Failed to activate .venv"
    return 1 2>/dev/null || exit 1
  fi
  echo "VIRTUAL_ENV=${VIRTUAL_ENV?}"


  #echo "📋 Syncing dependencies from pyproject.toml..."
  if [ "$HAS_NOTEBOOK" -ne 1 ]; then
    echo "📋 Syncing base dev dependencies (no notebook)..."
    if ! uv sync --dev; then
      echo "❌ Failed to sync base dev dependencies" >&2
      { return 1 2>/dev/null; } || exit 1
    fi
  else
    echo "📋 Syncing dev and notebook dependencies..."
    if ! uv sync --dev --group notebook; then
      echo "❌ Failed to sync notebook dependencies" >&2
      { return 1 2>/dev/null; } || exit 1
    fi
  fi

  # install editable python code, but only if src/<name> exists
  if [ -n "$PROJECT_NAME" ] && [ -d "$PROJECT_ROOT/src/$PROJECT_NAME" ]; then
    echo "🔧 Installing project in editable mode..."
    if ! uv pip install -e .; then
      echo "❌ Failed to install editable project" >&2
      { return 1 2>/dev/null; } || exit 1
    fi
  else
    echo "ℹ️  No src/$PROJECT_NAME directory found; skipping editable install."
  fi

  # register jupyter kernel, but only if notebook in pyproject.toml
  if [ "$HAS_NOTEBOOK" -eq 1 ]; then
    #echo "🧹 Removing default 'python3' kernel if present..."
    #jupyter kernelspec remove -y python3 || true

    KERNEL_NAME="$(basename "$PROJECT_ROOT")"
    echo "🧠 Registering Jupyter kernel: ${KERNEL_NAME?}..."
    if ! uv run ipython kernel install \
          --user \
          --env VIRTUAL_ENV "$PROJECT_ROOT/.venv" \
          --name "${KERNEL_NAME?}" \
          --display-name "${KERNEL_NAME?}"
    then
      echo "❌ Kernel registration failed — possibly missing ipykernel or misconfigured .venv" >&2
      echo "   Try: uv sync --dev --group notebook" >&2
      { return 1 2>/dev/null; } || true  # prevent killing ssh if sourced
    fi
    echo "Note: kernels persist outside of uv in '~/.local/share/jupyter/kernels/'"
    echo "Launch Jupyter with:"
    echo "  uv run --with notebook jupyter lab"
    echo "and select '${KERNEL_NAME?}' kernel"
  else
    echo "ℹ️  No notebook group detected — no kernel to add or register."
  fi

  echo "✅ Dev environment ready"

    # Explicitly clear traps on success (EXIT will also run, but this makes it obvious)
  trap - ERR
  trap - EXIT
  eval "$old_opts"

}

dev_init "$@"
