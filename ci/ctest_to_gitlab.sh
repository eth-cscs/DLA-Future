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
THREADS_PER_NODE="$3"
SLURM_CONSTRAINT="$4"

if [ "$USE_CODECOV" = true ]; then
BASE_TEMPLATE="
include:
  - remote: 'https://gitlab.com/cscs-ci/recipes/-/raw/master/templates/v2/.cscs.yml'

image: $IMAGE

stages:
  - test
  - upload

variables:
  SLURM_EXCLUSIVE: ''
  SLURM_EXACT: ''
  SLURM_CONSTRAINT: $SLURM_CONSTRAINT
  CRAY_CUDA_MPS: 1
  MPICH_MAX_THREAD_SAFETY: multiple

{{JOBS}}

upload_reports:
  stage: upload
  extends: .daint
  variables:
    PULL_IMAGE: 'NO'
    SLURM_NTASKS: 1
    SLURM_TIMELIMIT: '5:00'
    DISABLE_AFTER_SCRIPT: 'YES'
  script: upload_codecov
"
JOB_TEMPLATE="

{{CATEGORY_LABEL}}_{{RANK_LABEL}}:
  stage: test
  extends: .daint
  variables:
    SLURM_CPUS_PER_TASK: {{CPUS_PER_TASK}}
    SLURM_NTASKS: {{NTASKS}}
    SLURM_TIMELIMIT: '25:00'
    SLURM_UNBUFFEREDIO: 1
    SLURM_WAIT: 0
    PULL_IMAGE: 'YES'
    USE_MPI: 'YES'
    DISABLE_AFTER_SCRIPT: 'YES'
    DLAF_HDF5_TEST_OUTPUT_PATH: \$CI_PROJECT_DIR
  script: mpi-ctest -L {{CATEGORY_LABEL}} -L {{RANK_LABEL}}
  artifacts:
    paths:
      - codecov-reports/"
else
BASE_TEMPLATE="
include:
  - remote: 'https://gitlab.com/cscs-ci/recipes/-/raw/master/templates/v2/.cscs.yml'

image: $IMAGE

stages:
  - test

variables:
  SLURM_EXCLUSIVE: ''
  SLURM_EXACT: ''
  SLURM_CONSTRAINT: $SLURM_CONSTRAINT
  CRAY_CUDA_MPS: 1
  MPICH_MAX_THREAD_SAFETY: multiple

{{JOBS}}
"

JOB_TEMPLATE="
{{CATEGORY_LABEL}}_{{RANK_LABEL}}:
  stage: test
  extends: .daint
  variables:
    SLURM_CPUS_PER_TASK: {{CPUS_PER_TASK}}
    SLURM_NTASKS: {{NTASKS}}
    SLURM_TIMELIMIT: '20:00'
    SLURM_UNBUFFEREDIO: 1
    SLURM_WAIT: 0
    PULL_IMAGE: 'YES'
    USE_MPI: 'YES'
    DISABLE_AFTER_SCRIPT: 'YES'
    DLAF_HDF5_TEST_OUTPUT_PATH: \$CI_PROJECT_DIR
  script: mpi-ctest -L {{CATEGORY_LABEL}} -L {{RANK_LABEL}}"
fi

JOBS=""

for rank_label in `ctest --print-labels | egrep -o "RANK_[1-9][0-9]?"`; do
    for category_label in `ctest --print-labels | egrep -o "CATEGORY_[A-Z]+"`; do
        N=`echo "$rank_label" | sed "s/RANK_//"`
        C=$(( THREADS_PER_NODE / N ))

        # Skip label combinations that match no tests
        if ! ctest -N -L $category_label -L $rank_label | egrep --quiet "^Total Tests: [1-9][0-9]?$"; then
            continue
        fi

        JOB=`echo "$JOB_TEMPLATE" | sed "s|{{CATEGORY_LABEL}}|$category_label|g" \
                                  | sed "s|{{RANK_LABEL}}|$rank_label|g" \
                                  | sed "s|{{NTASKS}}|$N|g" \
                                  | sed "s|{{CPUS_PER_TASK}}|$C|g"`

        JOBS="$JOBS$JOB"
    done
done

echo "${BASE_TEMPLATE/'{{JOBS}}'/$JOBS}"
