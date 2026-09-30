#!/usr/bin/env bash
# Publish docs/wiki/*.md to this repository's GitHub wiki.
#
# Usage:
#   bash scripts/publish_wiki.sh ["commit message"]
#
# Requires push access to https://github.com/<owner>/<repo>.wiki.git
# (HTTPS credential helper / token, or SSH). The wiki must already be
# initialized: open https://github.com/<owner>/<repo>/wiki, create the first
# page (e.g. Home), and save - GitHub only creates the .wiki.git repository
# after that.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

ORIGIN="$(git remote get-url origin)"
WIKI_URL="${ORIGIN%.git}.wiki.git"
MESSAGE="${1:-docs: sync wiki pages from docs/wiki}"

WORKDIR="$(mktemp -d)"
trap 'rm -rf "${WORKDIR}"' EXIT

echo "==> cloning ${WIKI_URL}"
if ! git clone --depth 1 "${WIKI_URL}" "${WORKDIR}"; then
  echo "error: could not clone the wiki repository." >&2
  echo "       Enable the wiki (Settings -> Features -> Wikis) and create the" >&2
  echo "       first page in the web UI, then retry." >&2
  exit 1
fi

cp "${ROOT}"/docs/wiki/*.md "${WORKDIR}/"
cd "${WORKDIR}"

git add --all
if git diff --cached --quiet; then
  echo "==> wiki already up to date"
  exit 0
fi
git commit -m "${MESSAGE}"
git push
echo "==> published ${WIKI_URL}"
