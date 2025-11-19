#!/bin/bash

# Used to get a specific tool version from the pyproject.toml file
#   it can be a python version, or in a tool section
# Used by:
#   1. update-pre-commit-versions-from-pyproject.toml: to update .pre-commit-config.yaml
#   2. extract-tool-version.yaml: determine version needed for github action CI/CD steps

# Extract package version (python only) as resolved by uv show
extract-tool-version-from-pyproject-toml-from-uv() {

    # Note:
    #   * This will show the *installed* version, even if it differs
    #     from pyproject.toml, so other methods that look directly
    #     at pyproject.toml are preferred since it will not match
    #     if user forgets to uv sync after git pull
    #   * uv pip show produces multi-line output, so keep the line with
    #     "Version: <version>", and discard the "Version: " with the \K
    #   * This should not have surrounding quotes

    tool_name=${1?}
    pyproject_toml_path=${2:-pyproject.toml}
    pyproject_toml_dir=$(dirname ${pyproject_toml_path?})

    echo "Searching for ${tool_name?} version from pyrpoject.toml, using 'uv pip show' (uv project dir ${pyproject_toml_dir?})" >&2
    uv --project ${pyproject_toml_dir?} pip show ${tool_name?} | grep -Po "Version: \K.*"
}


# Search for package version (python only) using a grep regex on requirements
extract-tool-version-from-pyproject-toml-from-requirements() {

    # Notes:
    #   * will find pip-requirements-style packages
    #       <package> (==|<=|<|>=) <version>
    #   * the whole line or the individual <version> and <package> may be quoted
    #   * we don't want to allow '<' or '>' version, but '==', '<=', '>=' are ok
    #   * ensure it is of the form #.#.#...

    tool_name=${1?}
    pyproject_toml_path=${2:-pyproject.toml}

    echo "Searching for ${tool_name?} version using pyproject.toml, using grep for python requirements (${pyproject_toml_path?})" >&2
    grep -Eo "\s*\"?${tool_name?}\"?\s*(=|<|>)*=\s*\"?[0-9]+(\.[0-9]+)*\"?,?" ${pyproject_toml_path?} | grep -Eo "[0-9]+(\.[0-9]+)*"
}


# Search for package version (including non-python) using grep on tool sections
extract-tool-version-from-pyproject-toml-from-section() {

    # Search for  a non-python tool in pyproject toml.  It must be of the form
    #   [tool.<name>]
    #   version = <version>

    # Notes:
    #   * the first part of the regex searches multiple lines within
    #     a single section until the word "version" is found following
    #     whitespace (treating whole file as a single line with the -z grep option)
    #   * the next part returns everything after the =\s* until the end of the
    #     line with the \K regex "keep after" operator
    #   * the trim command removes any null characters from the grep -z option
    #   * the final step removes any quotation marks that may surround the version
    #

    tool_name=${1?}
    pyproject_toml_path=${2:-pyproject.toml}

    echo "Searching for ${tool_name?} version from pyproject.toml, using grep for tool sections (${pyproject_toml_path?})" >&2
    grep -Pzo "\[tool\.${tool_name?}\][^\[]*\n\s*version\s*=\s*\K[^\n]+" ${pyproject_toml_path?} | tr -d '\0' | sed 's/[\"'\'']//g'
}


get-project-root-dir() {
    # Check if the current directory contains /scripts/
    if [[ "$PWD" =~ /dev-scripts/?$ ]]; then
      # If so, set root_dir to the parent directory
      root_dir=$(dirname "$PWD")
    # Check if the current directory contains /scripts/
    elif [[ "$PWD" =~ /scripts/?$ ]]; then
      # If so, set root_dir to the parent directory
      root_dir=$(dirname "$PWD")
    elif [ -f "$PWD/pyproject.toml" ]; then
      # Otherwise, set root_dir to the current directory
      root_dir="$PWD"
    else
        echo "Root directory not found, must have pyproject.toml and neither dev-scripts nor scripts in path" >&2
        exit 1
    fi
    echo "Found root directory at ${root_dir?}" >&2
    echo ${root_dir?}
}


# Extract package version from pyproject.toml all methods (even non-python)
extract-tool-version-from-pyproject-toml-all() {
    # Search for tool (including non-python) in the following order:
    #   1. using uv: `uv pip show`
    #   2. in pyproject.toml requirements:
    #           looking for `<tool> (==|>=|<=) <version>`
    #   3. in pyproject.toml [tool.section]:
    #           looking for following line `<tool> = <version>

    tool_name=${1?}

    root_dir=$(get-project-root-dir)
    pyproject_toml_path="$root_dir/pyproject.toml"
    echo "Extracting $tool_name version from pyproject.toml, path to pyproject.toml used: $pyproject_toml_path" >&2

    if [ -f "$pyproject_toml_path" ]; then

        # Note: using uv sync will extract the installed version, even if it differs from
        # pyproject.toml.  This will be bad if user forgets to uv sync after git pull, but
        # we don't want anything here to trigger a uv sync
        #version=$(extract-tool-version-from-pyproject-toml-from-uv $tool_name $pyproject_toml_path)

        version=$(extract-tool-version-from-pyproject-toml-from-requirements $tool_name $pyproject_toml_path)

        if [ $? -ne 0 ]; then
            version=$(extract-tool-version-from-pyproject-toml-from-section $tool_name $pyproject_toml_path)
        fi

        if [ $? -ne 0 ]; then
            echo "Tool $tool_name not found in pyproject.toml" >&2
            exit 1
        fi

        echo "Tool $tool_name found in pyproject.toml, version: $version" >&2
        echo $version

    else

        echo "File pyproject.toml not found in PWD=$PWD" >&2
        exit 1

    fi
}


# Check if the script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  # Script is being executed directly
  extract-tool-version-from-pyproject-toml-all ${1?}
fi
