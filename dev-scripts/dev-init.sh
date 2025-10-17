#!/bin/bash

# Prep dev environment
# 1. check if sourcing
# 2. check for uv
# 3. check for pyproject.toml
# 4. check for active venv that matches project_root (not other repo)
# 5. create venv if it doesn't exist
# 6. activate venv if not already active
# 7. register environment as jupyter kernel

# This script must be sourced, so these checks are invalid
# shellcheck disable=SC2317,SC1091

# Ensure script is sourced, not executed
(return 0 2>/dev/null) || {
  echo "❌ This script must be sourced, not executed."
  echo "   Run it like this:  source dev-scripts/dev-init.sh"
  exit 1
}

# Resolve directory of this script, then its parent (the project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR?}/.." && pwd)"
cd "${PROJECT_ROOT?}" || {
  echo "❌ Could not cd to $PROJECT_ROOT" >&2
  return 1 2>/dev/null || true
}


# for safesync call
source "${PROJECT_ROOT?}/dev-scripts/safe-sync.sh"
source "${PROJECT_ROOT?}/dev-scripts/register-jupyter-kernel.sh"

dev_init() {

  # Temporarily disable interactive history & file appends to avoid pollution
  set +o history

  # Save current options
  local old_opts
  old_opts=$(set +o)

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

  # handle errors
  trap '
    set -- "$?"  # $1 = original status (avoids SC2154 on a named var)
    # disarm immediately so we do not re-enter / double-report
    trap - ERR
    # try to show actual line and name
    echo "❌ Error in ${BASH_SOURCE[1]:-${BASH_SOURCE[0]}} at line ${BASH_LINENO[0]:-$LINENO}"
    # Return original status if we can
    return "$1" 2>/dev/null || true
  ' ERR

  # cleanup (restore) on return
  trap '
    # Self-disarm first; this handler should only run once
    # Also disarm ERR in case anything else fails below or later
    trap - RETURN ERR
    # disable nounset before touching old_opts
    set +u
    # Restore saved shell options if they were captured
    [ "${old_opts+x}" ] && eval "$old_opts"
    # Re-enable history
    set -o history
  ' RETURN


  # Check for pyproject.toml
  PYPROJECT_FILE="${PROJECT_ROOT?}/pyproject.toml"
  if [ ! -f "${PYPROJECT_FILE?}" ]; then
    echo "❌ No pyproject.toml found in $PROJECT_ROOT, expected a minimal skeleton" >&2
    { return 1 2>/dev/null; } || exit 1
  fi

  # -- Sync ---
  # Synchronize, creating venv environment if needed
  # call it and propagate errors
  safe_sync || { st=$?; echo "[dev-init] safe_sync failed (status $st)" >&2; return "$st"; }

  # --- Activation phase ---
  #
  echo "⚡ Activating venv (if needed)"
  local REPO_VENV="${PROJECT_ROOT?}/.venv"
  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    # If a venv is already active, it must be the project .venv
    if [ "$(readlink -f -- "$VIRTUAL_ENV")" != "$(readlink -f -- "$REPO_VENV")" ]; then
      echo "❌ Active VIRTUAL_ENV differs from project .venv" >&2
      echo "    active: $VIRTUAL_ENV" >&2
      echo "    expect: $REPO_VENV" >&2
      { return 1 2>/dev/null; } || exit 1
    else
      echo "...🟢 Already active: $VIRTUAL_ENV"
    fi
  else
    if ! source "${REPO_VENV?}/bin/activate"; then
      echo "❌ Failed to activate $REPO_VENV" >&2
      { return 1 2>/dev/null; } || exit 1
    fi
    echo "...✅ Activated: $VIRTUAL_ENV"
  fi

  # --- Register jupyter notebook ---
  register_jupyter_kernel || { st=$?; echo "[dev-init] register_jupyter_kernel failed (status $st)" >&2; return "$st"; }


  # ---- Done ----

  echo "✅ Dev environment ready"
  # Note: trap will restore old opts and history recording

}

dev_init "$@"

# Defensive: if RETURN didn’t fire (due to return inside ERR trap),
# don’t leave traps behind
trap - ERR RETURN
