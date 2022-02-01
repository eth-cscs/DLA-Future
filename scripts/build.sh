#!/bin/bash

# Template build script
# ---------------------

rm -rf CMakeCache.txt CMakeFiles

export CC=<FIXME: path to C compiler>
export CXX=<FIXME: path to C++ compiler>
export MKLROOT=<FIXME: path to the root installation directory of MKL>
pika_DIR=<FIXME: path to the root installation directory of pika>
blaspp_DIR=<FIXME: path to the root installation directory of BLASPP>
lapackpp_DIR=<FIXME: path to the root installation directory of LAPACKPP>

# See README.md for all available options
cmake <FIXME: path to DLA-Future src directory> \
  -D DLAF_WITH_MKL=ON \
  -D CMAKE_BUILD_TYPE=RelWithDebugInfo \

make <FIXME: targets to build>
