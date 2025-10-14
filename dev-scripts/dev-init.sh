#!/bin/bash

# Prep dev environment
# 1. check if sourcing
# 2. check for uv
# 3. check for pyproject.toml
# 4. check for active venv that matches project_root (not other repo)
# 5. create venv if it doesn't exist
# 6. activate venv if not already active
# 7. install current project as an editable package

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
  
  # Temporarily disable interactive history & file appends to avoid pollution
  set +o history

  # Save current options
  local old_opts
  old_opts=$(set +o)

  # Safely remove traps
  # Always restore options and history recording when this function exits
  # Note: will handle even if old_opts is unset (which can leave stale
  #       traps in bash otherwise)
  trap 'set +u; trap - RETURN ERR; [ "${old_opts+x}" ] && eval "$old_opts"; set -o history' RETURN

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
    echo "❌ Could not cd to $PROJECT_ROOT" >&2
    return 1 2>/dev/null || true
  }

  # Check for pyproject.toml
  PYPROJECT_FILE="${PROJECT_ROOT?}/pyproject.toml"
  if [ ! -f "$PYPROJECT_FILE" ]; then
    echo "❌ No pyproject.toml found in $PROJECT_ROOT, expected a minimal skeleton" >&2
    { return 1 2>/dev/null; } || exit 1
  fi

  # Check for active environment for other repo
  if [ -n "${VIRTUAL_ENV:-}" ]; then
    # If a venv is already active, it must be the project .venv
    if [ "$(readlink -f -- "$VIRTUAL_ENV")" != "$(readlink -f -- "$REPO_VENV")" ]; then
      echo "❌ Active VIRTUAL_ENV differs from project .venv" >&2
      echo "    active: $VIRTUAL_ENV" >&2
      echo "    expect: $REPO_VENV" >&2
      { return 1 2>/dev/null; } || exit 1
    fi
  fi
  
  # --- Environment creation phase ---

  echo "📦 Creating virtual environment..."
  if [ -d "$REPO_VENV" ]; then
    echo "...♻️  Virtual environment already exists and valid — reusing: $REPO_VENV"
  else
    echo "...🆕 Creating new .venv at: $REPO_VENV"
    if ! uv venv; then
      echo "❌ Failed to create .venv" >&2
      { return 1 2>/dev/null; } || exit 1
    fi
  fi

  # --- Activation phase ---
  echo "⚡ Activating virtual environment..."
  if [ -n "${VIRTUAL_ENV:-}" ]; then
    echo "...🟢 Virtual environment already activated: ${VIRTUAL_ENV}"
  else
    if ! source "$REPO_VENV/bin/activate"; then
      echo "❌ Failed to activate .venv" >&2
      { return 1 2>/dev/null; } || exit 1
    fi
    echo "...✅ Virtual environment activated: ${VIRTUAL_ENV}"
  fi

  # --- Install editable ---

  echo "🔧 Installing project in editable mode..."
  if ! uv pip install -e .; then
    echo "❌ Failed to install editable project" >&2
    { return 1 2>/dev/null; } || exit 1
  else
    echo "...✅ Editable project installed"
  fi

  echo "✅ Dev environment ready"

  # Note: trap will restore old opts and history recording

}

dev_init "$@"
