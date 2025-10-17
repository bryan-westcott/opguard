#!/bin/bash

# Register jupyter kernel with same name as project
# but only if there is a 'notebook' group in pyproject.toml

# Note: this has to be a subshell to avoid conflicting with caller traps
register_jupyter_kernel() {

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

    # register jupyter kernel, but only if notebook in pyproject.toml
    if [ "${HAS_NOTEBOOK_GROUP?}" -eq 1 ]; then
      KERNEL_NAME="$(basename "${PROJECT_ROOT?}")"
      echo "🧠 Registering Jupyter kernel: ${KERNEL_NAME?}..."
      if ! uv run ipython kernel install \
            --user \
            --env VIRTUAL_ENV "${PROJECT_ROOT?}/.venv" \
            --name "${KERNEL_NAME?}" \
            --display-name "${KERNEL_NAME?}"
      then
        echo "❌ Kernel registration failed — possibly missing ipykernel or misconfigured .venv" >&2
        echo "   Try: uv sync --dev --group notebook" >&2
        exit 1
      fi
      echo "Warning: kernels persist outside of uv in '~/.local/share/jupyter/kernels/'"
      echo "Warning: kernels may collide if same project is initialized elsewhere"
      echo "Launch with: "
      echo "    uv run --with notebook jupyter lab"
      echo "Choose: '${KERNEL_NAME?}' kernel"
      echo "Warning: jupyter bug may require click in file browser before kernel can be selected"
      echo "...✅ Registered jupyter kernel ${KERNEL_NAME?}"
    else
      echo "...ℹ️  No notebook group detected — no kernel to add or register."
    fi

  # close subshell
  )
  # return subshell's status to the caller
  local st=$?
  return "$st"
}

# Only run if executed directly, not when sourced:
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    register_jupyter_kernel "$@" || exit
fi
