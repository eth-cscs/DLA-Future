#!/bin/bash -l
#SBATCH --job-name=DLA-Future-compile-mc
#SBATCH --nodes=1
#SBATCH --constraint=mc
#SBATCH --partition=cscsci
#SBATCH --time=00:15:00

set -e

source ${ENV}
module list

set -x
DIR=build_${BUILD_TYPE}

# If exists remove previous build
rm -rf $DIR

mkdir -p $DIR
cd $DIR
cmake -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
      -DDLAF_WITH_MKL=1                \
      -DHPX_DIR=$EBROOTHPX             \
      -Dblaspp_DIR=$EBROOTBLASPP       \
      -Dlapackpp_DIR=$EBROOTLAPACKPP   \
      -DDLAF_MPI_PRESET=slurm          \
      -DMPIEXEC_NUMCORES=36            \
      -DDLAF_CI_TWEAK=1                \
      -DCMAKE_CXX_FLAGS="-Werror"      \
      ../
make -j 20 VERBOSE=1

# only executed if build passed
echo "compilation successful"
