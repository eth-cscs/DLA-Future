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

# make sure no test log exists
rm -f Testing/Temporary/LastTest.log*

set +e
make test
RETCODE=$?
set -e

# add test log to output
cat Testing/Temporary/LastTest.log*

(exit $RETCODE)

# only executed if tests passed
echo "Test Successful"
cd ..
rm -rf $DIR
