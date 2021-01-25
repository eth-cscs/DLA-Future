#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2021, ETH Zurich
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

  set(IS_COMPILER_CLANG $<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>)
  set(IS_COMPILER_GCC $<CXX_COMPILER_ID:GCC>)

  target_compile_options(${target_name}
    PRIVATE
      -Wall
      -Wextra
      -Wnon-virtual-dtor
      -Wunused
      -Woverloaded-virtual
      -Wdangling-else
      -Wswitch-enum
      # Conversions
      $<${IS_COMPILER_GCC}:
        -Wsign-conversion
        -Wfloat-conversion>
      $<${IS_COMPILER_CLANG}:
        -Wbitfield-enum-conversion
        -Wbool-conversion
        -Wconstant-conversion
        -Wenum-conversion
        -Wfloat-conversion
        -Wint-conversion
        -Wliteral-conversion
        -Wnon-literal-null-conversion
        -Wnull-conversion
        -Wshorten-64-to-32
        -Wsign-conversion
        -Wstring-conversion>
      -pedantic-errors

      # googletest macro problem
      # must specify at least one argument for '...' parameter of variadic macro
      $<${IS_COMPILER_CLANG}:-Wno-gnu-zero-variadic-macro-arguments>
    )
endmacro()
