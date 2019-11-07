#!/bin/bash

# Template build script
# ---------------------

rm -rf CMakeCache.txt CMakeFiles

export CC=<FIXME: path to C compiler>
export CXX=<FIXME: path to C++ compiler>
export MKLROOT=<FIXME: path to the root installation directory of MKL>
HPX_DIR=<FIXME: path to the root installation directory of HPX>
BLASPP_DIR=<FIXME: path to the root installation directory of BLASPP>
LAPACKPP_DIR=<FIXME: path to the root installation directory of LAPACKPP>

# CMAKE_BUILD_TYPE := Release | RelWithDebugInfo | Debug
#
# DLAF_WITH_CUDA   := ON | OFF (default)
#
# DLAF_WITH_MKL    := ON | OFF (default)
#
# BUILD_DOC        := ON | OFF (default)
#
# DLAF_WITH_TEST   := ON (default) | OFF
#
cmake <FIXME: path to DLA-Future src directory> \
  -D DLAF_WITH_MKL=ON \
  -D CMAKE_BUILD_TYPE=RelWithDebugInfo \

make <FIXME: targets to build>
