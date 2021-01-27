#!/bin/bash -l
#SBATCH --job-name=DLA-Future-build-doc
#SBATCH --nodes=1
#SBATCH --constraint=mc
#SBATCH --partition=cscsci
#SBATCH --time=00:15:00
#SBATCH --output=make-doc.out.txt
#SBATCH --error=make-doc.out.txt

set -e

source ${ENV}
module list

set -x
DIR=build_doc

# If exists remove previous build
rm -rf $DIR

mkdir -p $DIR
cd $DIR
cmake -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
      -DDLAF_WITH_MKL=1                \
      -DHPX_DIR=$EBROOTHPX             \
      -Dblaspp_DIR=$EBROOTBLASPP       \
      -Dlapackpp_DIR=$EBROOTLAPACKPP   \
      -DDLAF_BUILD_DOC=1               \
      ../
make doc -j 50 VERBOSE=1

# only executed if build passed
echo "doc generated"
cd ..
rm -rf $DIR
