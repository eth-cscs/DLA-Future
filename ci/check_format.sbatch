#!/bin/bash -l
#SBATCH --job-name=DLA-Future-build-doc
#SBATCH --nodes=1
#SBATCH --constraint=mc
#SBATCH --partition=cscsci
#SBATCH --time=00:05:00
#SBATCH --output=check_format.out.txt
#SBATCH --error=check_format.out.txt

set -e

module purge
module load modules
module load craype
module load cray-mpich
module load slurm
module load xalt
module load PrgEnv-gnu
module load daint-mc
module use /apps/daint/SSL/software/spack-current/share/spack/modules/cray-cnl6-broadwell
module load llvm/8.0.0-gcc-6.2.0-cray-cnl6-broadwell-release

git config --add remote.origin.fetch +refs/heads/master:refs/remotes/origin/master
git fetch --no-tags

set -x

FILES_CHANGED=`git diff --name-only origin/master`

for FILE in $FILES_CHANGED
do
  case $FILE in
    *.cpp|*.h|*.hpp|*.ipp)
      clang-format -i --style=file $FILE
      ;;
    *.pdf|*.hdf5|*.jpg|*.png|*.ppt|*.pptx)
      # Exclude some binary files types,
      # others can be excluded as needed.
      ;;
    *)
      # Remove trailing whitespaces.
      sed -i "s/\\s\\+$//g" $FILE

      # Count number of trailing newlines at the end of the file.
      # For performance reason we limit the search (max is 32).
      n_newlines=`tail -c 32 $FILE | xxd -p -c 32 | egrep -o "(0a)+$" | grep -o 0a | wc -l`

      # Add or remove trailing newlines if needed.
      if [[ $n_newlines -eq 0 ]]
      then
        echo "" >> $FILE
      elif [[ $n_newlines -gt 1 ]]
      then
        truncate -s -$(($n_newlines-1)) $FILE
      fi
      ;;
  esac
done
# Fails if there are differences.
git diff --exit-code
