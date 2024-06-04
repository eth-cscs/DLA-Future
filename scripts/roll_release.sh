#!/usr/bin/env bash

#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

# This script tags a release locally and creates a release on GitHub. It relies
# on the GitHub CLI (https://cli.github.com).

set -o errexit

REPO="eth-cscs/DLA-Future"
VERSION_MAJOR=$(sed -n 's/project(DLAF VERSION \([0-9]\+\)\.[0-9]\+\.[0-9]\+)/\1/p' CMakeLists.txt)
VERSION_MINOR=$(sed -n 's/project(DLAF VERSION [0-9]\+\.\([0-9]\+\)\.[0-9]\+)/\1/p' CMakeLists.txt)
VERSION_PATCH=$(sed -n 's/project(DLAF VERSION [0-9]\+\.[0-9]\+\.\([0-9]\+\))/\1/p' CMakeLists.txt)
VERSION_FULL="${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}"
VERSION_FULL_TAG="v${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}"
VERSION_TITLE="DLA-Future ${VERSION_FULL}"
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
RELEASE_DATE=$(date '+%Y-%m-%d')

if ! which gh >/dev/null 2>&1; then
    echo "GitHub CLI not installed on this system (see https://cli.github.com). Exiting."
    exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
    echo 'gh is not logged in. Run `gh auth login` to authenticate with your GitHub account, or set `GITHUB_TOKEN` to a token with `public_repo` access. Exiting.'
    exit 1
fi

# Major and minor releases are made directly from master. Patch releases are branched out from the major
# and minor releases with a version_X.Y branch.
if [[ "${VERSION_PATCH}" -eq 0 ]]; then
    RELEASE_BRANCH="master"
else
    RELEASE_BRANCH="version_${VERSION_MAJOR}.${VERSION_MINOR}"
fi

if ! [[ "$CURRENT_BRANCH" == "$RELEASE_BRANCH" ]]; then
    echo "Not on release branch (expected \"$RELEASE_BRANCH\", currently on \"${CURRENT_BRANCH}\"). Not continuing to make release."
    exit 1
fi

changelog_path="CHANGELOG.md"
readme_path="README.md"
cff_path="CITATION.cff"

echo "You are about to tag and create a final release on GitHub."

echo ""
echo "Sanity checking release"

sanity_errors=0

printf "Checking that the git repository is in a clean state... "
if [[ $(git status --porcelain | wc -l) -eq 0  ]] ; then
    echo "OK"
else
    echo "ERROR"
    sanity_errors=$((sanity_errors + 1))
fi

printf "Checking that %s has an entry for %s... " "${changelog_path}" "${VERSION_FULL}"
if grep "## DLA-Future ${VERSION_FULL}" "${changelog_path}"; then
    echo "OK"
else
    echo "Missing"
    sanity_errors=$((sanity_errors + 1))
fi

printf "Checking that %s has a documentation entry for %s... " "${readme_path}" "${VERSION_FULL}"
if grep "^- \[Documentation of \`${VERSION_FULL_TAG}\`\](https://eth-cscs.github.io/DLA-Future/${VERSION_FULL_TAG}" "${readme_path}"; then
    echo "OK"
else
    echo "Missing"
    sanity_errors=$((sanity_errors + 1))
fi

printf "Checking that %s has correct version for %s... " "${cff_path}" "${VERSION_FULL}"
if grep "^version: ${VERSION_FULL}" "${cff_path}"; then
    echo "OK"
else
    echo "Missing"
    sanity_errors=$((sanity_errors + 1))
fi

printf "Checking that %s has correct title for %s... " "${cff_path}" "${VERSION_FULL}"
if grep "^title: ${VERSION_TITLE}" "${cff_path}"; then
    echo "OK"
else
    echo "Missing"
    sanity_errors=$((sanity_errors + 1))
fi

printf "Checking that %s has today's date... " "${cff_path}"
if grep "^date-released: '${RELEASE_DATE}'" "${cff_path}"; then
    echo "OK"
else
    echo "Missing"
    sanity_errors=$((sanity_errors + 1))
fi

if [[ ${sanity_errors} -gt 0 ]]; then
    echo "Found ${sanity_errors} error(s). Fix it/them and try again."
    exit 1
fi

# Extract the changelog for this version
VERSION_DESCRIPTION=$(
    # Find the correct heading and print everything from there to the end of the file
    awk "/^## DLA-Future ${VERSION_FULL}/,EOF" ${changelog_path} |
        # Remove the heading
        tail -n+3 |
        # Find the next heading or the end of the file and print everything until that heading
        sed '/^## /Q' |
        # Move headings one level up, i.e. transform ### to ##, ## to #, etc. There should be no
        # top-level heading in the file except for "# Changelog".
        sed 's/^##/#/'
)

echo ""
echo "The version is: ${VERSION_FULL}"
echo "The version title is: ${VERSION_TITLE}"
echo "The release date is: ${RELEASE_DATE}"
echo "The version description is:"
echo "${VERSION_DESCRIPTION}"

echo "Do you want to continue?"
select yn in "Yes" "No"; do
    case $yn in
    Yes) break ;;
    No) exit ;;
    esac
done

echo ""
if [[ "$(git tag -l ${VERSION_FULL_TAG})" == "${VERSION_FULL_TAG}" ]]; then
    echo "Tag already exists locally."
else
    echo "Tagging release."
    git tag --annotate "${VERSION_FULL_TAG}" --message="${VERSION_TITLE}"
fi

remote=$(git remote -v | grep "github.com[:/]eth-cscs/DLA-Future.git" | cut -f1 | uniq)
if [[ "$(git ls-remote --tags --refs $remote | grep -o ${VERSION_FULL_TAG})" == "${VERSION_FULL_TAG}" ]]; then
    echo "Tag already exists remotely."
else
    echo "Pushing tag to $remote."
    git push $remote "${VERSION_FULL_TAG}"
fi

echo ""
echo "Creating release."
gh release create "${VERSION_FULL_TAG}" \
    --repo "${REPO}" \
    --title "${VERSION_TITLE}" \
    --notes "${VERSION_DESCRIPTION}"
