#!/bin/bash

# Prep dev environment
# Safely uv sync based on sections found

# Note: this has to be a subshell to avoid conflicting with caller traps
safe_sync() {

  # run in a subshell
  (
    # Temporarily (subshell only) disable interactive history & file appends to avoid pollution
    set +o history
    set -euo pipefail

    # Resolve directory of this script, then its parent (the project root)
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "${SCRIPT_DIR?}/.." && pwd)"
    cd "${PROJECT_ROOT?}" || {
      echo "❌ Could not cd to $PROJECT_ROOT" >&2
      exit 1
    }

    # Check for pyproject.toml
    PYPROJECT_FILE="${PROJECT_ROOT?}/pyproject.toml"
    if [ ! -f "${PYPROJECT_FILE?}" ]; then
      echo "❌ No pyproject.toml found in $PROJECT_ROOT, expected a minimal skeleton" >&2
      exit 1
    fi

    # -- Check for notebook ---

    # check if notebook section in pyproject.toml
    HAS_NOTEBOOK_GROUP=0
    if grep -q '^[[:space:]]*notebook[[:space:]]*=' "${PYPROJECT_FILE?}"; then
      echo "📓 Detected 'notebook' group in $PYPROJECT_FILE"
      HAS_NOTEBOOK_GROUP=1
    else
      echo "🚫📓 No 'notebook' group found in $PYPROJECT_FILE."
    fi

    # -- Check for tests ---

    # check if tests section in pyproject.toml
    # NOTE: this should be an EXTRA not a group!
    HAS_TEST_EXTRA=0
    if grep -q '^[[:space:]]*test[[:space:]]*=' "${PYPROJECT_FILE?}"; then
      echo "🔬 Detected 'tests' group in $PYPROJECT_FILE"
      HAS_TEST_EXTRA=1
    else
      echo "🚫🔬  No 'tests' group found in $PYPROJECT_FILE."
    fi

    # -- Check for tooling ---

    # check if tooling section in pyproject.toml
    HAS_TOOLING_GROUP=0
    if grep -q '^[[:space:]]*tooling[[:space:]]*=' "${PYPROJECT_FILE?}"; then
      echo "🛠️  Detected 'tooling' extra in $PYPROJECT_FILE"
      HAS_TOOLING_GROUP=1
    else
      echo "🚫🛠️  No 'tooling' extra found in $PYPROJECT_FILE; skipping Jupyter setup."
    fi

    # -- Compose sync args ---

    # One sync pass to populate venv (frozen if lock exists)
    UV_SYNC_ARGS=()
    # Make dev explicit
    UV_SYNC_ARGS+=(--dev)
    # For pre-commit/linting/CI/CD
    [[ "${HAS_TOOLING_GROUP?}" -eq 1 ]] && UV_SYNC_ARGS+=(--group tooling)
    # For notebook support
    [[ "${HAS_NOTEBOOK_GROUP?}" -eq 1 ]] && UV_SYNC_ARGS+=(--group notebook)
    # For tests
    # Note: an optional group not dependency group
    [[ "${HAS_TEST_EXTRA?}" -eq 1 ]] && UV_SYNC_ARGS+=(--extra test)
    # Run sync
    echo "🔄 Syncing with args: ${UV_SYNC_ARGS[*]}"
    uv sync "${UV_SYNC_ARGS[@]}"
    echo "...✅ Sync successful"

  # close subshell
  )
  # return subshell's status to the caller
  local st=$?
  return "$st"
}

# Only run if executed directly, not when sourced:
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    safe_sync "$@" || exit
fi
