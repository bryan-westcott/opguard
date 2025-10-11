#!/usr/bin/env bash
set -e

file="$1"
if [[ -z "$file" ]]; then
  echo "Usage: $0 <file>" >&2
  exit 1
fi

repo_root=$(git rev-parse --show-toplevel)

case "$file" in
  *.tex) mode=tex ;;
  *.md)  mode=markdown ;;
  *.txt) mode=none ;;
  *)     mode=none ;;
esac

aspell --mode="$mode" \
       --home-dir="$repo_root" \
       --personal="$repo_root/.aspell.en.pws" \
       check "$file"
