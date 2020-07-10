#!/bin/bash -l

set -eux

commit_status=${1}

curl --verbose \
    --request POST \
    --url "https://api.github.com/repos/eth-cscs/DLA-Future/statuses/${CI_COMMIT_SHA}" \
    --header 'Content-Type: application/json' \
    --header "authorization: Bearer ${GITHUB_TOKEN}" \
    --data "{ \"state\": \"${commit_status}\", \"target_url\": \"${CI_PIPELINE_URL}\", \"description\": \"Gitlab\", \"context\": \"gitlab-status\" }"