#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

if(NOT DLAF_BUILD_TESTING_HEADER)
  return()
endif()

add_custom_target(test_header)

function(DLAF_addHeaderTest header)
  # Create a filename with a cpp/cu extension based on the header file. This
  # assumes a relative header path.
  get_filename_component(header_ext "${header}" EXT)
  if(DLAF_WITH_GPU AND ("${header_ext}" STREQUAL ".cu.h"))
    string(REGEX REPLACE "\.h$" "" test_source "${header}")
    if(DLAF_WITH_HIP)
      set_source_files_properties("${test_source}" LANGUAGE HIP)
    endif()
  else()
    string(REGEX REPLACE "\.h$" ".cpp" test_source "${header}")
  endif()
  set(test_source "${CMAKE_CURRENT_BINARY_DIR}/${test_source}")

  # Create a test name
  string(REGEX REPLACE "[^A-Za-z]" "_" test_name "${header}")
  set(test_name "test_header_${test_name}")

  # Write a source file which will only include a single header for testing that
  # the header is self-contained
  file(WRITE "${test_source}" "#include <${header}>")

  add_library(${test_name} OBJECT "${test_source}")
  target_link_libraries(${test_name} PRIVATE dlaf.prop dlaf.prop_private)
  target_add_warnings(${test_name})
  add_dependencies(test_header ${test_name})
endfunction()

file(GLOB_RECURSE headers RELATIVE "${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/include/*.h")
foreach(header ${headers})
  dlaf_addheadertest(${header})
endforeach()
