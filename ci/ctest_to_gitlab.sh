#!/bin/bash

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
  - allocate
{{TEST_STAGES}}
  - upload
  - cleanup

variables:
  ALLOCATION_NAME: dlaf-ci-job-\$CI_PIPELINE_ID
  SLURM_EXCLUSIVE: ''
  SLURM_CONSTRAINT: $SLURM_CONSTRAINT
  CRAY_CUDA_MPS: 1

allocate:
  stage: allocate
  extends: .daint_alloc
  variables:
    PULL_IMAGE: 'YES'
    SLURM_TIMELIMIT: '15:00'

{{JOBS}}

upload_reports:
  stage: upload
  extends: .daint
  variables:
    PULL_IMAGE: 'NO'
    SLURM_NTASKS: 1
    SLURM_TIMELIMIT: '15:00'
    DISABLE_AFTER_SCRIPT: 'YES'
  script: upload_codecov

deallocate:
  stage: cleanup
  extends: .daint_dealloc
"
JOB_TEMPLATE="

{{LABEL}}:
  stage: test_{{LABEL}}
  extends: .daint
  variables:
    SLURM_CPUS_PER_TASK: {{CPUS_PER_TASK}}
    SLURM_NTASKS: {{NTASKS}}
    SLURM_TIMELIMIT: '15:00'
    PULL_IMAGE: 'NO'
    USE_MPI: 'YES'
    DISABLE_AFTER_SCRIPT: 'YES'
  script: mpi-ctest -L {{LABEL}}
  artifacts:
    paths:
      - codecov-reports/"
else
BASE_TEMPLATE="
include:
  - remote: 'https://gitlab.com/cscs-ci/recipes/-/raw/master/templates/v2/.cscs.yml'

image: $IMAGE

stages:
  - allocate
{{TEST_STAGES}}
  - cleanup

variables:
  ALLOCATION_NAME: dlaf-ci-job-\$CI_PIPELINE_ID
  SLURM_EXCLUSIVE: ''
  SLURM_CONSTRAINT: $SLURM_CONSTRAINT
  CRAY_CUDA_MPS: 1

allocate:
  stage: allocate
  extends: .daint_alloc
  variables:
    PULL_IMAGE: 'YES'
    SLURM_TIMELIMIT: '15:00'

{{JOBS}}

deallocate:
  stage: cleanup
  extends: .daint_dealloc
"

JOB_TEMPLATE="
{{LABEL}}:
  stage: test_{{LABEL}}
  extends: .daint
  variables:
    SLURM_CPUS_PER_TASK: {{CPUS_PER_TASK}}
    SLURM_NTASKS: {{NTASKS}}
    SLURM_TIMELIMIT: '15:00'
    PULL_IMAGE: 'NO'
    USE_MPI: 'YES'
    DISABLE_AFTER_SCRIPT: 'YES'
  script: mpi-ctest -L {{LABEL}}"
fi

TEST_STAGE_TEMPLATE="
  - test_{{LABEL}}
"

JOBS=""
TEST_STAGES=""

for label in `ctest --print-labels | egrep -o "RANK_[1-9][0-9]?"`; do
    N=`echo "$label" | sed "s/RANK_//"`
    C=$(( THREADS_PER_NODE / N ))

    JOB=`echo "$JOB_TEMPLATE" | sed "s|{{LABEL}}|$label|g" \
                              | sed "s|{{NTASKS}}|$N|g" \
                              | sed "s|{{CPUS_PER_TASK}}|$C|g"`
    TEST_STAGE=`echo "$TEST_STAGE_TEMPLATE" | sed "s|{{LABEL}}|$label|g"`

    JOBS="$JOBS$JOB"
    TEST_STAGES="$TEST_STAGES$TEST_STAGE"
done

TMP_JOB=`echo "${BASE_TEMPLATE/'{{JOBS}}'/$JOBS}"`
echo "${TMP_JOB/'{{TEST_STAGES}}'/$TEST_STAGES}"
