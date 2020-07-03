#!/bin/bash

CORES_PER_NODE=36

BASE_TEMPLATE="
include:
  - remote: 'https://gitlab.com/cscs-ci/recipes/-/raw/master/templates/v1/.cscs.yml'

image: $1

stages:
  - allocate
  - test
  - upload
  - cleanup

# Make one big allocation reused in all jobs
variables:
  ALLOCATION_NAME: dlaf-ci-job-\$CI_PIPELINE_ID
  SLURM_EXCLUSIVE: ''

# Allocate the resources
allocate:
  stage: allocate
  extends: .daint_alloc
  variables:
    PULL_IMAGE: 'YES'
    SLURM_TIMELIMIT: '15:00'

# Execute multiple jobs
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

# Remove the allocation
deallocate:
  stage: cleanup
  extends: .daint_dealloc
"

JOB_TEMPLATE="

{{LABEL}}:
  stage: test
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

JOBS=""

for label in `ctest --print-labels | egrep -o "RANK_[1-9][0-9]?"`; do
    N=`echo "$label" | sed "s/RANK_//"`
    C=$(( CORES_PER_NODE / N ))

    JOB=`echo "$JOB_TEMPLATE" | sed "s|{{LABEL}}|$label|g" \
                              | sed "s|{{NTASKS}}|$N|g" \
                              | sed "s|{{CPUS_PER_TASK}}|$C|g"`

    JOBS="$JOBS$JOB"
done

echo "${BASE_TEMPLATE/'{{JOBS}}'/$JOBS}"
