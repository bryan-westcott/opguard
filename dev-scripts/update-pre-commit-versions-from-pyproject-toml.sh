#!/bin/bash
# Update pre-commit using versions in pyproject.toml requirement or tool section

update_pre_commit_versions() {
    root_dir=$(pwd)
    local pyproject_toml_path="${root_dir?}/pyproject.toml"
    local pre_commit_config_yaml_path="${root_dir?}/.pre-commit-config.yaml"
    local tool_name=${1?}
    local repo_url=".${tool_name?}.*"

    version_requirements=$(grep -Eo "\s*\"?${tool_name?}\"?\s*(=|<|>)*=\s*\"?[0-9]+(\.[0-9]+)*\"?,?" ${pyproject_toml_path?} | grep -Eo "[0-9]+(\.[0-9]+)*")
    version_section=$(grep -Pzo "\[tool\.${tool_name?}\][^\[]*\n\s*version\s*=\s*\K[^\n]+" ${pyproject_toml_path?} | tr -d '\0' | sed 's/[\"'\'']//g')
    version="${version_requirements:-$version_section}"
    repo_url_escaped=${repo_url//\//\\/}
    sed -i "/repo:\s*${repo_url_escaped?}.*/!b;n;s/rev:\s\+\(v\?\).*/rev: \1${version?}/" "${pre_commit_config_yaml_path?}"

    echo "version from requirement: ${version_requirements?}"
    echo "version from section: ${version_section?}"
    echo "version used: ${version?}"
}

# Check if the script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    update_pre_commit_versions ${1?} ${2}
fi
