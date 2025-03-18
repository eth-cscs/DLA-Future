#!/bin/bash

#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#
# This script is used to manually run tests on hohgant.
# It implies that an allocation is already set up.
#
# usage:
# > ssh hohgant
# > salloc -N 1 -p amdgpu
# > ./run_amd_gpu.sh <image name>
#
# note:
# if the image has already been downloaded with sarus the last line can be replaced with
# > ./run_amd_gpu.sh <image name> OFF
#

IMAGE="$1"
PULL="$2"

if [ -z "$IMAGE" ]; then
  echo "Usage: run_test_ault.sh <IMAGE> [OFF]"
  echo "       IMAGE: deploy image name for amd gpu tests"
  echo "       OFF:   disable image pulling (useful when the image has already been pulled)"
  exit 1
fi

if [ -z "$SLURM_JOB_ID" ]; then
  echo "SLURM allocation not available please run"
  echo "> salloc -N 1 -p amdgpu"
  exit 1
fi

if [ "$PULL" != "OFF" ]; then
  echo "Note: credentials can be found in user settings on jfrog.svc.cscs.ch. Passwords are not accepted, only the token can be used."
  sarus pull --login $IMAGE
fi

DEVICES="--device=/dev/kfd:rw --device=/dev/dri/card0:rw --device=/dev/dri/card1:rw --device=/dev/dri/card2:rw --device=/dev/dri/card3:rw --device=/dev/dri/card4:rw --device=/dev/dri/card5:rw --device=/dev/dri/card6:rw --device=/dev/dri/card7:rw --device=/dev/dri/renderD128:rw --device=/dev/dri/renderD129:rw --device=/dev/dri/renderD130:rw --device=/dev/dri/renderD131:rw --device=/dev/dri/renderD132:rw --device=/dev/dri/renderD133:rw --device=/dev/dri/renderD134:rw --device=/dev/dri/renderD135:rw"

failed=0
for label in `sarus run $IMAGE ctest --print-labels | egrep -o "RANK_[1-9][0-9]?"`; do
  var=`sarus run $IMAGE ctest -N -V -L $label | egrep -o "Test command:.*$" | sed 's/^Test command: //'`
  for test_cmd in `echo "$var"`; do
    echo "Running: $label $test_cmd"
    srun -n `echo $label | egrep -o "[0-9]+"` -c 16 --mpi=pmi2 --cpu-bind=core sarus run $DEVICES $IMAGE bash -c "ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID `eval echo $test_cmd`"
    if [ $? != 0 ]; then
      failed=$(( failed + 1 ))
    fi
    sleep 1
  done
done

echo "-----------------------"
if [ $failed == 0 ]; then
  echo "All tests PASSED"
else
  echo "$failed tests FAILED!!!"
fi
