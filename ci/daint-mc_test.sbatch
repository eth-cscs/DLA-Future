#!/bin/bash -l
#SBATCH --job-name=DLA-Future-test-mc
#SBATCH --nodes=1
#SBATCH --constraint=mc
#SBATCH --partition=cscsci
#SBATCH --time=00:15:00

set -e

source ${ENV}
module list

set -x
DIR=build_${BUILD_TYPE}

cd $DIR
make test

# only executed if tests passed
echo "test successful"
cd ..
rm -rf $DIR
