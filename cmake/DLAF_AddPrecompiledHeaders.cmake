#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2022, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

function(DLAF_addPrecompiledHeaders target_name)
  if(DLAF_WITH_PRECOMPILED_HEADERS)
    get_target_property(target_type ${target_name} TYPE)
    if(target_type STREQUAL "EXECUTABLE")
      target_precompile_headers(${target_name} REUSE_FROM dlaf.pch_exe)
    else()
      target_precompile_headers(${target_name} REUSE_FROM dlaf.pch_lib)
    endif()
  endif()
endfunction()
