#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2019, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

# Try compiling (not linking) the C++ code passed in and
# returns if it is possible to build it or not
function (try_compile_cxx_code code built)
  execute_process(
    COMMAND echo "${code}"
    COMMAND ${CMAKE_CXX_COMPILER} -c -x c++ -o- -
    OUTPUT_QUIET
    ERROR_QUIET
    RESULT_VARIABLE result)

  if (NOT result)
    set(${built} TRUE PARENT_SCOPE)
  else()
    set(${built} FALSE PARENT_SCOPE)
  endif()
endfunction()
