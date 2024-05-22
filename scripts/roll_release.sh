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
# on the hub command line tool (https://hub.github.com/).

set -o errexit

VERSION_MAJOR=$(sed -n 's/project(DLAF VERSION \([0-9]\+\)\.[0-9]\+\.[0-9]\+)/\1/p' CMakeLists.txt)
VERSION_MINOR=$(sed -n 's/project(DLAF VERSION [0-9]\+\.\([0-9]\+\)\.[0-9]\+)/\1/p' CMakeLists.txt)
VERSION_PATCH=$(sed -n 's/project(DLAF VERSION [0-9]\+\.[0-9]\+\.\([0-9]\+\))/\1/p' CMakeLists.txt)
VERSION_FULL="${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}"
VERSION_FULL_TAG="v${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}"
VERSION_TITLE="DLA-Future ${VERSION_FULL}"
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"

if ! which hub >/dev/null 2>&1; then
    echo "Hub not installed on this system (see https://hub.github.com/). Exiting."
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
if git status --porcelain ; then
    echo "ERROR"
    sanity_errors=$((sanity_errors + 1))
else
    echo "OK"
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
echo "The version description is:"
echo "${VERSION_DESCRIPTION}"

echo "Do you want to continue?"
select yn in "Yes" "No"; do
    case $yn in
    Yes) break ;;
    No) exit ;;
    esac
done

if [[ -z "${GITHUB_USER}" || -z "${GITHUB_PASSWORD}" ]] && [[ -z "${GITHUB_TOKEN}" ]]; then
    echo "Need GITHUB_USER and GITHUB_PASSWORD or only GITHUB_TOKEN to be set to use hub release."
    exit 1
fi

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
hub release create \
    --message="${VERSION_TITLE}" \
    --message="${VERSION_DESCRIPTION}" \
    "${VERSION_FULL_TAG}"
