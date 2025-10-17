#!/bin/bash
#------------------------------------------------------------------------------
# update-pre-commit-versions.sh
#
# Purpose
#   Update the `rev:` pin for a given tool's repository in `.pre-commit-config.yaml`
#   using the version discovered in `pyproject.toml`. It looks in two places:
#     1) As a requirement-style version (e.g., in `[project.dependencies]`
#        or similar), extracting the numeric version for the given tool name.
#     2) In a tool-specific section, e.g. `[tool.<name>]`, reading a `version = "..."`
#        line.
#
# How it works (high-level)
#   - Reads the current working directory as the repo root.
#   - Greps `pyproject.toml` for a version that matches the provided tool name.
#   - If not found in requirement-style entries, looks under `[tool.<name>]` for a
#     `version = ...` line.
#   - Constructs a regex for the tool's repo URL pattern expected in pre-commit
#     (default pattern is `.<tool>.*`, which matches URLs containing `.<tool>`)
#   - Uses `sed` to locate the matching `repo:` entry and then update the following
#     `rev:` line to the discovered version, preserving an optional leading `v`.
#
# Assumptions / Expectations
#   - `.pre-commit-config.yaml` has entries like:
#         - repo: <something ending in .<tool>...>
#           rev: vX.Y.Z
#     and that `rev:` is on the *next* line after the `repo:` line.
#   - The repo URL match pattern is deliberately broad: `.<tool>.*`
#     (e.g., matches `https://github.com/psf/black` when tool is `black` because
#     of `.black` in the path). Adjust if your repos use different naming.
#   - `pyproject.toml` contains either:
#       - a requirement-like spec including `<tool>` and a bare numeric version, or
#       - a `[tool.<tool>]` table with a `version = "X.Y.Z"` entry.
#
# Requirements
#   - GNU grep with PCRE support (uses `-P` and `-z`).
#   - sed with in-place editing (`-i`).
#   - tr, sed, grep standard behavior assumed on Linux/macOS.
#
# Usage
#   ./update-pre-commit-versions.sh <tool_name>
#
#   Examples:
#     ./update-pre-commit-versions.sh black
#     ./update-pre-commit-versions.sh ruff
#
# Exit codes
#   - Relies on `${1?}` to ensure a tool name is provided (shell will error if not).
#   - `sed` will quietly do nothing if it cannot find a matching `repo:` line
#     followed by a `rev:` line; inspect the final echo for the detected version.
#
# Caveats / Limitations
#   - If multiple `repo:` entries match `.<tool>.*`, sed will update the first match.
#   - If your `.pre-commit-config.yaml` is structured differently (e.g., `rev:` not
#     immediately after `repo:`), the update won’t occur.
#   - If versions in `pyproject.toml` are specified with complex constraints (e.g.,
#     `>=`, `~=`, extras), only the bare numeric version is extracted.
#   - BSD/macOS `sed -i` syntax differs when providing a backup suffix; this script
#     uses GNU-style `-i` without a suffix.
#
# Note: how to use with pre-commit autoupdate
#   - simply run `uv run pre-commit autoupdate`
#   - then a git diff will show which versions are stale
#   - update the stale versions in the `pyproject.toml` file and commit
#       both `pyproject.toml` and `.pre-commit-config.yaml`
#   - the pre-commit hook will pull them from there and compare with your
#     committed versions
#   - for the pyproj-template project, you will also need to update both
#     the base.* files which will become the future defaults.
#
#------------------------------------------------------------------------------

update_pre_commit_versions() {
    # Treat the current working directory as the repo root
    root_dir=$(pwd)
    local pyproject_toml_path="${root_dir?}/pyproject.toml"
    local pre_commit_config_yaml_path="${root_dir?}/.pre-commit-config.yaml"

    # Ensure both files exist
    if [[ ! -f "${pyproject_toml_path}" ]]; then
      echo "ERROR: missing ${pyproject_toml_path}" >&2
      return 1
    fi
    if [[ ! -f "${pre_commit_config_yaml_path}" ]]; then
      echo "ERROR: missing ${pre_commit_config_yaml_path}" >&2
      return 1
    fi

      # require exactly one non-empty arg
    if [[ $# -lt 1 || -z ${1-} ]]; then
      echo "Usage: update_pre_commit_versions <tool_name>" >&2
      return 2
    fi
    # Required parameter: tool name (e.g., "ruff")
    local tool_name=${1?}
    # also reject all-whitespace
    if [[ -z "${tool_name//[[:space:]]/}" ]]; then
      echo "ERROR: <tool_name> cannot be blank/whitespace" >&2
      return 2
    fi

    # --- 1) Detect version ----
    #
    # requirements-style versions (may yield 0..N lines)
    mapfile -t version_requirements_arr < <(
        grep -Eo "\s*\"?${tool_name?}\"?\s*(=|<|>)*=\s*\"?[0-9]+(\.[0-9]+)*\"?,?" "${pyproject_toml_path?}" \
        | grep -Eo "[0-9]+(\.[0-9]+)*" || true
    )

    # [tool.<name>] section version (may yield 0..N lines)
    mapfile -t version_section_arr < <(
        grep -Pzo "\[tool\.${tool_name?}\][^\[]*\n\s*version\s*=\s*\K[^\n]+" "${pyproject_toml_path?}" \
        | tr -d '\0' \
        | sed 's/[\"'\'']//g' || true
    )

    # If more than one requirements match, error
    if (( ${#version_requirements_arr[@]} > 1 )); then
        echo "ERROR: multiple requirement-style versions found for '${tool_name}': ${version_requirements_arr[*]}" >&2
        return 1
    fi
    # If more than one section match, error
    if (( ${#version_section_arr[@]} > 1 )); then
        echo "ERROR: multiple [tool.${tool_name}] versions found: ${version_section_arr[*]}" >&2
        return 1
    fi
    # If neither have any matches, error
    if (( ${#version_requirements_arr[@]} == 0 )) && (( ${#version_section_arr[@]} == 0 )); then
        echo "ERROR: no version found for '${tool_name}' in requirements or [tool.${tool_name}] section." >&2
        return 1
    fi
    # if both have matches, error
    if (( ${#version_requirements_arr[@]} == 1 )) && (( ${#version_section_arr[@]} == 1 )); then
        echo "ERROR: version found in BOTH requirements and [tool.${tool_name}] section (ambiguous)." >&2
        echo "       requirements=${version_requirements_arr[0]} section=${version_section_arr[0]}" >&2
        return 1
    fi
    # Prefer requirement hit, else section hit (after checks)
    local version="${version_requirements_arr[0]:-${version_section_arr[0]}}"


    # --- Detect repo_url ---

    # Pattern used to identify the target repo line in pre-commit config:
    # We expect something like: "repo: ...<something>.<tool_name>..."
    # The leading dot ensures we don't over-match short substrings (heuristic).
    local repo_url_pattern="repo:\s*.*${tool_name?}.*"
    local rev_pattern="rev:\s\+\(v\?\)"
    # Escape slashes for sed; we inject the repo/rev patterns into a sed script
    local repo_url_pattern_escaped=${repo_url_pattern//\//\\/}
    local rev_pattern_escaped=${rev_pattern//\//\\/}

    # Ensure exactly one repo match (same pattern as sed address)
    mapfile -t _repo_matches < <(
      sed -n "/${repo_url_pattern_escaped?}.*/p" "${pre_commit_config_yaml_path?}"
    )
    if [[ ${#_repo_matches[@]} -eq 0 ]]; then
      echo "ERROR: no matching 'repo:' for pattern '${repo_url_pattern}' in ${pre_commit_config_yaml_path}" >&2
      return 1
    fi
    if [[ ${#_repo_matches[@]} -gt 1 ]]; then
      echo "ERROR: multiple matching 'repo:' lines for pattern '.${tool_name}.*' in ${pre_commit_config_yaml_path}:" >&2
      printf '  - %s\n' "${_repo_matches[@]}" >&2
      return 1
    fi

    # Check for desired repo followed by "rev:" revision line
    sed -n "/${repo_url_pattern_escaped?}.*/!b; n; /${rev_pattern_escaped?}.*/!q1; q0" "${pre_commit_config_yaml_path?}" \
        || { echo "ERROR: expected 'repo: …${tool_name}…' immediately followed by a 'rev:' line"; exit 1; }

    # ------------------------------------------------------------
    # Update `.pre-commit-config.yaml`:
    #    - Find the line that matches `repo:\s*.<tool>.*`
    #    - On the *next* line, replace `rev:` with the discovered version,
    #      preserving an optional existing leading 'v' (with `\(v\?\)`).
    #
    #    sed script breakdown:
    #      /repo:\s*${repo_url_pattern_escaped}.*/   -> match the target repo line
    #      !b                                 -> if not matched, branch to end (no-op)
    #      n                                  -> move to the next line (expected `rev:`)
    #      s/rev:\s\+\(v\?\).*/rev: \1${version}/ -> substitute rev, keep optional 'v'
    # ------------------------------------------------------------
    sed -i "/${repo_url_pattern_escaped?}.*/!b; n; s/${rev_pattern_escaped?}.*/rev: \1${version?}/" "${pre_commit_config_yaml_path?}"

    # Now ensure the substitution was successful
    sed -n "/${repo_url_pattern_escaped?}.*/!b; n; /${rev_pattern_escaped?}.*/!q1; /${version?}/!q1; q0" "${pre_commit_config_yaml_path?}" \
        || { echo "ERROR: repo block not updated to version '${version}'"; exit 1; }

    # --- Done ---

    # Final status echo for visibility
    echo "tool=${tool_name}, version=${version}"
}

# ------------------------------------------------------------------------------
# Entry point
#   If the script file is executed directly, run the updater with the first arg.
#   If it's sourced, do nothing (function is then available in the shell).
# ------------------------------------------------------------------------------
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    update_pre_commit_versions "${1?}" "${2}"
fi
