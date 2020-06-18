#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2019, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

# DLAF_addTest(test_target_name
#   SOURCES <source1> [<source2> ...]
#   [COMPILE_DEFINITIONS <arguments for target_compile_definitions>]
#   [INCLUDE_DIRS <arguments for target_include_directories>]
#   [LIBRARIES <arguments for target_link_libraries>]
#   [MPIRANKS <number of rank>]
#   [USE_MAIN {PLAIN | HPX | MPI | MPIHPX}]
# )
#
# At least one source file has to be specified, while other parameters are optional.
#
# COMPILE_DEFINITIONS, INCLUDE_DIRS and LIBRARIES are passed to respective cmake wrappers, so it is
# possible to specify PRIVATE/INTERFACE/PUBLIC modifiers.
#
# MPIRANKS specifies the number of ranks on which the test will be carried out and it implies a link with
# MPI library. At build time the constant NUM_MPI_RANKS=MPIRANKS is set.
#
# USE_MAIN links to an external main function, in particular:
#   - PLAIN: uses the classic gtest_main
#   - HPX: uses a main that initializes HPX
#   - MPI: uses a main that initializes MPI
#   - MPIHPX: uses a main that initializes both HPX and MPI
# If not specified, no external main is used and it should exist in the test source code.
#
# e.g.
#
# DLAF_addTest(example_test
#   SOURCE main.cpp testfixture.cpp
#   LIBRARIES
#     PRIVATE
#       boost::boost
#       include/
# )

function(DLAF_addTest test_target_name)
  set(options "")
  set(oneValueArgs MPIRANKS USE_MAIN)
  set(multiValueArgs SOURCES COMPILE_DEFINITIONS INCLUDE_DIRS LIBRARIES ARGUMENTS)
  cmake_parse_arguments(DLAF_AT "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  ### Checks
  if (DLAF_AT_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "Unknown arguments ${DLAF_AT_UNPARSED_ARGUMENTS}")
  endif()

  if (NOT DLAF_AT_SOURCES)
    message(FATAL_ERROR "No sources specified for this test")
  endif()

  set(IS_AN_MPI_TEST FALSE)
  set(IS_AN_HPX_TEST FALSE)
  if (NOT DLAF_AT_USE_MAIN)
    set(_gtest_tgt gtest)
  elseif (DLAF_AT_USE_MAIN STREQUAL PLAIN)
    set(_gtest_tgt gtest_main)
  elseif (DLAF_AT_USE_MAIN STREQUAL HPX)
    set(_gtest_tgt DLAF_gtest_hpx_main)
    set(IS_AN_HPX_TEST TRUE)
  elseif (DLAF_AT_USE_MAIN STREQUAL MPI)
    set(_gtest_tgt DLAF_gtest_mpi_main)
    set(IS_AN_MPI_TEST TRUE)
  elseif (DLAF_AT_USE_MAIN STREQUAL MPIHPX)
    set(_gtest_tgt DLAF_gtest_mpihpx_main)
    set(IS_AN_MPI_TEST TRUE)
    set(IS_AN_HPX_TEST TRUE)
  else()
    message(FATAL_ERROR "USE_MAIN=${DLAF_AT_USE_MAIN} is not a supported option")
  endif()

  if (IS_AN_MPI_TEST)
    if (NOT DLAF_AT_MPIRANKS)
      message(FATAL_ERROR "You are asking for an MPI external main without specifying MPIRANKS")
    endif()
    if (NOT DLAF_AT_MPIRANKS GREATER 0)
      message(FATAL_ERROR "Wrong MPIRANKS number ${DLAF_AT_MPIRANKS}")
    endif()
    if (DLAF_AT_MPIRANKS GREATER MPIEXEC_MAX_NUMPROCS)
      message(WARNING "\
      YOU ARE ASKING FOR ${DLAF_AT_MPIRANKS} RANKS, BUT THERE ARE JUST ${MPIEXEC_MAX_NUMPROCS} CORES.
      You can adjust MPIEXEC_MAX_NUMPROCS value to suppress this warning.
      Using OpenMPI may require to set the environment variable OMPI_MCA_rmaps_base_oversubscribe=1.")
    endif()
  else()
    if (DLAF_AT_MPIRANKS)
      message(FATAL_ERROR "You specified MPIRANKS and asked for an external main without MPI")
    else()
      set(DLAF_AT_MPIRANKS 1)
    endif()
  endif()

  ### Test target
  set(DLAF_TEST_RUNALL_WITH_MPIEXEC OFF CACHE BOOL "Run all tests using the workload manager.")

  set(_TEST_ARGUMENTS ${DLAF_AT_ARGUMENTS})

  if(DLAF_TEST_RUNALL_WITH_MPIEXEC OR IS_AN_MPI_TEST)
    if (MPIEXEC_NUMCORE_FLAG)
      if (MPIEXEC_NUMCORES)
        set(_CORES_PER_RANK ${MPIEXEC_NUMCORES})
      else()
        set(_CORES_PER_RANK 1)
      endif()

      math(EXPR DLAF_CORE_PER_RANK "${_CORES_PER_RANK}/${DLAF_AT_MPIRANKS}")

      if (NOT DLAF_CORE_PER_RANK)
        set(DLAF_CORE_PER_RANK 1)
      endif()

      set(_MPI_CORE_ARGS ${MPIEXEC_NUMCORE_FLAG} ${DLAF_CORE_PER_RANK})
    else()
      set(_MPI_CORE_ARGS "")
    endif()

    if(DLAF_CI_RUNNER_USES_MPIRUN)
      set(_TEST_COMMAND $<TARGET_FILE:${test_target_name}>)
    else()
      set(_TEST_COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${DLAF_AT_MPIRANKS} ${_MPI_CORE_ARGS}
          ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${test_target_name}> ${MPIEXEC_POSTFLAGS})
    endif()
    set(_TEST_LABEL "RANK_${DLAF_AT_MPIRANKS}")

  # ----- Classic test
  else()
    set(_TEST_COMMAND ${test_target_name})
    set(_TEST_LABEL "RANK_1")
  endif()

  if (IS_AN_HPX_TEST)
    separate_arguments(_HPX_EXTRA_ARGS_LIST UNIX_COMMAND ${DLAF_HPXTEST_EXTRA_ARGS})

    # APPLE platform does not support thread binding
    if (NOT APPLE)
      list(APPEND _TEST_ARGUMENTS "--hpx:use-process-mask")
    endif()

    list(APPEND _TEST_ARGUMENTS ${_HPX_EXTRA_ARGS_LIST})
  endif()

  ### Test executable target
  add_executable(${test_target_name} ${DLAF_AT_SOURCES})
  target_link_libraries(${test_target_name}
    PRIVATE
      ${_gtest_tgt}
      DLAF_test
      ${DLAF_AT_LIBRARIES}
  )
  target_compile_definitions(${test_target_name}
    PRIVATE
      ${DLAF_AT_COMPILE_DEFINITIONS}
      $<$<BOOL:${IS_AN_MPI_TEST}>: NUM_MPI_RANKS=${DLAF_AT_MPIRANKS}>
  )
  target_include_directories(${test_target_name} PRIVATE ${DLAF_AT_INCLUDE_DIRS})
  target_add_warnings(${test_target_name})
  add_test(
    NAME ${test_target_name}
    COMMAND ${_TEST_COMMAND} ${_TEST_ARGUMENTS}
  )
  set_tests_properties(${test_target_name} PROPERTIES LABELS "${_TEST_LABEL}")

  ### DEPLOY
  include(GNUInstallDirs)

  set(DLAF_INSTALL_TESTS OFF CACHE BOOL "If tests are built, it controls if they will be installed")
  if (DLAF_INSTALL_TESTS)
    install(TARGETS
      ${test_target_name}
      # EXPORT DLAF-tests
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
  endif()
endfunction()
