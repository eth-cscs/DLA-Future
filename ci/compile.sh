#!/bin/bash -x
# Usage ./compile.sh <system> <build type>
# system: daint-mc
# build type: Release, Debug

export SYSTEM=$1
export BUILD_TYPE=$2

export BASE=${SYSTEM}_compile-${BUILD_TYPE}
export OUT=${BASE}.out.txt

echo "----- $SYSTEM compile $BUILD_TYPE -----"
export ENV=`realpath ci/${SYSTEM}_env.sh`

sbatch -o ${OUT} -e ${OUT} --wait ci/${SYSTEM}_compile.sbatch
