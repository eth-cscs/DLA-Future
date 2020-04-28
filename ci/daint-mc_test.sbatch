#!/bin/bash -l
#SBATCH --job-name=DLA-Future-test-mc
#SBATCH --nodes=1
#SBATCH --constraint=mc
#SBATCH --partition=cscsci
#SBATCH --time=00:15:00

set -e

# Utilities for timers
## current time in seconds
function ct { date +"%s"; }
## et(t) elapsed time since instant t (got with ct command)
function et { date +%T -d "1/1 + $(( `ct` - $1 )) sec"; }

source ${ENV}
module list

DIR=build_${BUILD_TYPE}

cd $DIR

# make sure no test log exists
rm -f Testing/Temporary/Test.log
rm -f Testing/Temporary/LastTest.log*

set +e

RETCODE=0
FULL_START=$(ct)
# Allow 1-99 ranks
for label in `ctest --print-labels | egrep -o "RANK_[1-9][0-9]?"`
do
  N=`echo $label | sed "s/RANK_//"`
  C=$(( CORES_PER_NODE / N ))
  CMD="ctest -L $label"

  # prepare run configuration
  echo "0 $CMD" > run.conf
  if [ "$N" -ge 2 ]
  then
    # run non master rank quietly
    echo "1-$(( N - 1 )) $CMD -Q" >> run.conf
  fi

  TEST_START=$(ct)
  echo "- Running: srun -n $N -c $C $CMD"
  srun -n $N -c $C --multi-prog run.conf
  TMP=$?
  if [ "$RETCODE" -eq 0 ]; then RETCODE=$TMP; fi
  echo "- Elapsed: $(et $TEST_START)"

  # append test log to combined test log and delete it
  cat Testing/Temporary/LastTest.log* >> Testing/Temporary/Test.log
  rm -f Testing/Temporary/LastTest.log*
done
echo "--- Total Elapsed: $(et $FULL_START)"

set -e

# add test log to output
cat Testing/Temporary/Test.log

(exit $RETCODE)

# only executed if tests passed
echo "Test Successful"
cd ..
rm -rf $DIR
