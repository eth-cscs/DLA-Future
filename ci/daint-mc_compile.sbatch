#!/bin/bash -l
#SBATCH --job-name=DLA-Future-compile-mc
#SBATCH --nodes=1
#SBATCH --constraint=mc
#SBATCH --partition=cscsci
#SBATCH --time=00:15:00

set -e

printenv
source ${ENV}
module list

set -x
DIR=build_${BUILD_TYPE}

# If exists remove previous build
rm -rf $DIR

mkdir -p $DIR
cd $DIR
cmake -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
      -DDLAF_USE_MKL=1                 \
      -DHPX_DIR=$EBROOTHPX             \
      -DBLASPP_ROOT=$EBROOTBLASPP      \
      -DLAPACKPP_ROOT=$EBROOTLAPACKPP  \
      -DMPIEXEC_EXECUTABLE="srun"      \
      -DMPIEXEC_NUMPROC_FLAG="-n "     \
      ../
make -j 50 VERBOSE=1

# only executed if build passed
echo "compilation successful"
