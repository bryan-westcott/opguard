#!/bin/bash

set -e

# Add current scripts directory to path
BASH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATH="$PATH:${BASH_DIR}"

# Source the other scripts using absolute paths
source extract-tool-version-from-pyproject-toml.sh

update_pre_commit() {
    local search_name=${1?}
    local repo_url=${2:-".*$search_name.*"}

    local root_dir
    root_dir=$(get-project-root-dir)

    local version
    version=$(extract-tool-version-from-pyproject-toml-all $search_name)

    if [ $? -ne 0 ]; then
        echo "Problem extracting version for ${search_name?}, exiting"
        exit 1
    fi

    if [ -n "${version?}" ]; then
        echo "Version for ${search_name?} found: ${version?}, updating pre-commit-config with repo_url: ${repo_url?}"
        # strip off any surrounding quote marks
        version=$(echo ${version?} | sed 's/[\"'\'']//g')
        # escape any forward slashes in the url to avoid issues with sed
        repo_url_escaped=${repo_url//\//\\/}
        # now search and replace
        # if an existing v is there, it will keep it
        sed -i "/repo:\s*${repo_url_escaped?}.*/!b;n;s/rev:\s\+\(v\?\).*/rev: \1${version?}/" "${root_dir?}/.pre-commit-config.yaml"
    else
        echo "No version found for ${search_name?}, skipping pre-commit update."
    fi
}


# Check if the script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    update_pre_commit ${1?} ${2}
fi
