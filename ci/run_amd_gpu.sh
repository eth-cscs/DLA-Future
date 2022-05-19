#!/bin/bash

#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2022, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#
# This script is used to manually run tests on ault.
# It implies that an allocation is already set up
# and that the correct GPUs are visible.
# E.g. for ault08: export HIP_VISIBLE_DEVICES=1

IMAGE="$1"
PULL="$2"

if [ -z "$IMAGE" ]; then
  echo "Usage: run_test_ault.sh <IMAGE> [OFF]"
  echo "       IMAGE: deploy image name for amd gpu tests"
  echo "       OFF:   disable image pulling (useful when the image has already been pulled)"
fi

if [ "$PULL" != "OFF" ]; then
  echo "Note: credentials can be found in user settings on art.cscs.ch. Passwords are not accepted, only the token can be used."
  sarus pull --login $IMAGE
fi

failed=0
for label in `sarus run $IMAGE ctest --print-labels | egrep -o "RANK_[1-9][0-9]?"`; do
  var=`sarus run $IMAGE ctest -N -V -L $label | egrep -o "Test command:.*$" | sed 's/^Test command: //'`
  while IFS= read -r test_cmd; do
    echo "Running: $label $test_cmd"
    srun -n `echo $label | egrep -o "[0-9]+"` -c 10 -i none sarus run --mount=type=bind,src=/dev/kfd,dst=/dev/kfd --mount=type=bind,src=/dev/dri,dst=/dev/dri $IMAGE `eval echo $test_cmd`
    if [ $? != 0 ]; then
      failed=$(( failed + 1 ))
    fi
  done <<< $"$var"
done

echo "-----------------------"
if [ $failed == 0 ]; then
  echo "All tests PASSED"
else
  echo "$failed tests FAILED!!!"
fi
