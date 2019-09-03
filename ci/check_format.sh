#!/bin/bash -l
module use /apps/daint/SSL/software/spack-current/share/spack/modules/cray-cnl6-broadwell
module load llvm/8.0.0-gcc-6.2.0-cray-cnl6-broadwell-release

git config --add remote.origin.fetch +refs/heads/master:refs/remotes/origin/master
git fetch --no-tags

set -ex

FILES_CHANGED=`git diff --name-only origin/master`

for FILE in $FILES_CHANGED
do
  case $FILE in
    *.cpp|*.h|*.hpp)
      clang-format -i --style=file $FILE
      ;;
    *)
      # remove trailing whitespaces
      sed -i "s/\\s\\+$//g" $FILE
      ;;
  esac
done
# Fails if there are differences.
git diff --exit-code
