#
# NS3C
#
# Copyright (c) 2018-2019, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

# NS3C_ADD_TEST(test_target_name
#   SOURCES <source1> [<source2> ...]
#   [COMPILE_DEFINITIONS <arguments for target_compile_definitions>]
#   [INCLUDE_DIRS <arguments for target_include_directories>]
#   [LIBRARIES <arguments for target_link_libraries>]
#   [MPIRANKS <number of rank>]
# )
#
# At least one source file has to be specified, while other parameters are optional.
# 
# COMPILE_DEFINITIONS, INCLUDE_DIRS and LIBRARIES are passed to respective cmake wrappers, so it is
# possible to specify PRIVATE/INTERFACE/PUBLIC modifiers.
# 
# MPIRANKS specifies the number of ranks on which the test will be carried out and it implies a link with
# MPI library.
# 
# e.g.
# 
# NS3C_ADD_TEST(example_test
#   SOURCE main.cpp testfixture.cpp
#   LIBRARIES
#     PRIVATE
#       boost::boost
#       include/
# )

function(NS3C_ADD_TEST test_target_name)
  set(options "USE_GTEST_MAIN")
  set(oneValueArgs MPIRANKS)
  set(multiValueArgs SOURCES COMPILE_DEFINITIONS INCLUDE_DIRS LIBRARIES ARGUMENTS)
  cmake_parse_arguments(NS3C_AT "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  ### Checks
  if (NS3C_AT_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "Unknown arguments ${NS3C_AT_UNPARSED_ARGUMENTS}")
  endif()

  if (NOT NS3C_AT_SOURCES)
    message(FATAL_ERROR "No sources specified for this test")
  endif()

  ### Test executable target
  add_executable(${test_target_name} ${NS3C_AT_SOURCES})

  if (NS3C_AT_COMPILE_DEFINITIONS)
    target_compile_definitions(${test_target_name}
      PRIVATE
        ${NS3C_AT_COMPILE_DEFINITIONS}
    )
  endif()

  if (NS3C_AT_INCLUDE_DIRS)
    target_include_directories(${test_target_name}
      ${NS3C_AT_INCLUDE_DIRS}
    )
  endif()
  
  if (NS3C_AT_USE_GTEST_MAIN)
    target_link_libraries(${test_target_name} PRIVATE gtest-main)
  else()
    target_link_libraries(${test_target_name} PRIVATE gtest)
  endif()

  if (NS3C_AT_LIBRARIES)
    target_link_libraries(${test_target_name}
      ${NS3C_AT_LIBRARIES}
    )
  endif()

  ### Test target
  # ----- MPI-based test
  if(DEFINED NS3C_AT_MPIRANKS)
    if (NOT NS3C_AT_MPIRANKS GREATER 0)
      message(FATAL_ERROR "Wrong MPIRANKS number ${NS3C_AT_MPIRANKS}")
    endif()

    if (NS3C_AT_MPIRANKS GREATER MPIEXEC_MAX_NUMPROCS)
      message(FATAL_ERROR "Impossible to have more than ${MPIEXEC_MAX_NUMPROCS}")
    endif()
    
    target_link_libraries(${test_target_name}
      PRIVATE MPI::MPI_CXX
    )
    
    add_test(
      NAME ${test_target_name}
      COMMAND
        ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${NS3C_AT_MPIRANKS}
        ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${test_target_name}> ${MPIEXEC_POSTFLAGS} ${NS3C_AT_ARGUMENTS})
  # ----- Classic test
  else()
    add_test(
      NAME ${test_target_name}
      COMMAND ${test_target_name} ${NS3C_AT_ARGUMENTS}
    )
  endif()
endfunction()
