#!/bin/bash -l
#SBATCH --job-name=DLA-Future-build-doc
#SBATCH --nodes=1
#SBATCH --constraint=mc
#SBATCH --partition=cscsci
#SBATCH --time=00:15:00
#SBATCH --output=make-doc.out.txt
#SBATCH --error=make-doc.out.txt

set -ex

source ${ENV}
module list
DIR=build_doc

mkdir -p $DIR
(
    cd $DIR
    cmake -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
          -DDLAF_USE_MKL=1                 \
          -DHPX_DIR=$EBROOTHPX             \
          -DBLASPP_ROOT=$EBROOTBLASPP      \
          -DLAPACKPP_ROOT=$EBROOTLAPACKPP  \
          -DBUILD_DOC=1                    \
          ../
    make doc -j 50 VERBOSE=1
) && echo "doc generated"
