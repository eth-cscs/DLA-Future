#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2019, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

macro(target_add_warnings target_name)
  if(NOT TARGET ${target_name})
    message(SEND_ERROR
      "${target_name} is not a target."
      "Cannot add warnings to it.")
  endif()

  target_compile_options(${target_name}
    PRIVATE
      -Wall
      -Wextra
      -Wnon-virtual-dtor
      -Wunused
      -Woverloaded-virtual
      -Wconversion
      -pedantic-errors

      # googletest macro problem
      # must specify at least one argument for '...' parameter of variadic macro
      -Wno-gnu-zero-variadic-macro-arguments
    )
endmacro()
