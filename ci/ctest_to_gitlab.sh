#!/bin/bash

#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

IMAGE="$1"
USE_CODECOV="$2"
THREADS_MAX_PER_TASK="$3"
THREADS_PER_NODE="$4"
SLURM_CONSTRAINT="$5"
RUNNER="$6"

if [ "$USE_CODECOV" = true ]; then
STAGES="
  - test
  - upload
"
EXTRA_JOBS="
upload_reports:
  stage: upload
  extends: $RUNNER
  variables:
    PULL_IMAGE: 'NO'
    SLURM_NTASKS: 1
    SLURM_TIMELIMIT: '5:00'
    DISABLE_AFTER_SCRIPT: 'YES'
  script: upload_codecov
"
TIMELIMIT="60:00"
ARTIFACTS="
  artifacts:
    when: always
    paths:
      - codecov-reports/
      - output/
"
else
STAGES="
  - test
"
EXTRA_JOBS=""
TIMELIMIT="45:00"
ARTIFACTS="
  artifacts:
    when: always
    paths:
      - output/
"
fi

BASE_TEMPLATE="
include:
  - remote: 'https://gitlab.com/cscs-ci/recipes/-/raw/master/templates/v2/.ci-ext.yml'
  - local: 'ci/ci-ext-custom.yml'

image: $IMAGE

stages:
$STAGES

variables:
  FF_TIMESTAMPS: true
  SLURM_EXCLUSIVE: ''
  SLURM_EXACT: ''
  SLURM_CONSTRAINT: $SLURM_CONSTRAINT
  CRAY_CUDA_MPS: 1
  MPICH_MAX_THREAD_SAFETY: multiple

{{JOBS}}

$EXTRA_JOBS

"

JOB_TEMPLATE="
{{CATEGORY_LABEL_NOPREFIX}}_{{RANK_LABEL}}:
  stage: test
  extends: $RUNNER
  variables:
    SLURM_CPUS_PER_TASK: {{CPUS_PER_TASK}}
    SLURM_NTASKS: {{NTASKS}}
    SLURM_TIMELIMIT: '$TIMELIMIT'
    SLURM_UNBUFFEREDIO: 1
    SLURM_WAIT: 0
    PULL_IMAGE: 'YES'
    USE_MPI: 'YES'
    DISABLE_AFTER_SCRIPT: 'YES'
    DLAF_HDF5_TEST_OUTPUT_PATH: /dev/shm
  script: stdbuf --output=L --error=L mpi-ctest -L {{CATEGORY_LABEL}} -L {{RANK_LABEL}}
  $ARTIFACTS
"

JOBS=""

for rank_label in `ctest --print-labels | egrep -o "RANK_[1-9][0-9]?"`; do
    for category_label in `ctest --print-labels | egrep -o "CATEGORY_[A-Z]+"`; do
        N=`echo "$rank_label" | sed "s/RANK_//"`
        C=$(( THREADS_PER_NODE / N ))
        if [ $C -gt $THREADS_MAX_PER_TASK ]; then
            C=$THREADS_MAX_PER_TASK
        fi

        # Skip label combinations that match no tests
        if [[ "$(ctest -N -L $category_label -L $rank_label | tail -n1)" == "Total Tests: 0" ]]; then
            continue
        fi

        category_label_noprefix=`echo "$category_label" | sed "s/CATEGORY_//"`
        JOB=`echo "$JOB_TEMPLATE" | sed "s|{{CATEGORY_LABEL_NOPREFIX}}|$category_label_noprefix|g" \
                                  | sed "s|{{CATEGORY_LABEL}}|$category_label|g" \
                                  | sed "s|{{RANK_LABEL}}|$rank_label|g" \
                                  | sed "s|{{NTASKS}}|$N|g" \
                                  | sed "s|{{CPUS_PER_TASK}}|$C|g"`

        JOBS="$JOBS$JOB"
    done
done

echo "${BASE_TEMPLATE/'{{JOBS}}'/$JOBS}"
