#!/bin/bash -i

# Setup pre-commit hooks on local dev environment, including required
# tools that may come from python, node, apt or compiled rust
#
# Note: tested on ubuntu 22.04

# Add current scripts directory to path
BASH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATH="$PATH:${BASH_DIR}"
# Also add parallel scripts directory
# (BASH_DIR is dev-scripts when running locally)
PATH="$PATH:${BASH_DIR}/../scripts"

# some tools come from uv
uv sync --dev --group notebook --frozen

# node-based tools
sudo apt install npm
prettier_version=$(extract-tool-version-from-pyproject-toml.sh prettier)
npm install --save-dev --save-exact "prettier@${prettier_version}"
action_validator_version=$(extract-tool-version-from-pyproject-toml.sh action-validator)
npm install --save-dev --save-exact "@action-validator/core@${action_validator_version}" "@action-validator/cli@${action_validator_version}"

# apt-based tools
sudo apt install yamllint shellcheck

# install latest rust for some tools
sudo apt remove rustc cargo
curl --proto '=https' --tlsv1.3 https://sh.rustup.rs -sSf | sh

# reload the bashrc
# WARNING: this will only work if shebang has the -i option due to ubuntu
source $HOME/.bashrc
# Install rust toolchain
rustup toolchain install stable

# Install golang for yamlfmt
sudo apt install golang

# install pre-commit and hooks as a uv tool (avoid need for uv run)
# Newer versions of pre-commit will break mypy hook
uv tool install "pre-commit==4.3.0" --with pre-commit-uv
pre-commit clean && pre-commit install --install-hooks

# test run
pre-commit run -a
