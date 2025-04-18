#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

set(DLAF_PIKATEST_EXTRA_ARGS "" CACHE STRING "Extra arguments for tests with pika")

macro(dlaf_setup_mpi_preset)
  set(MPIEXEC_EXECUTABLE_DESCRIPTION "Executable for running MPI programs")
  set(MPIEXEC_NUMPROC_FLAG_DESCRIPTION
      "Flag used by MPI to specify the number of processes for mpiexec; "
      "the next option will be the number of processes."
  )
  set(MPIEXEC_NUMCORE_FLAG_DESCRIPTION
      "Flag used by MPI to specify the number of cores per rank for mpiexec. "
      "If not empty, you have to specify also the number of cores available per rank in MPIEXEC_NUMCORES_PER_RANK."
  )
  set(MPIEXEC_NUMCORES_PER_RANK_DESCRIPTION "Number of cores used by each MPI rank.")

  # if a preset has been selected and it has been changed from previous configurations
  if(NOT DLAF_MPI_PRESET STREQUAL _DLAF_MPI_PRESET)

    if(DLAF_MPI_PRESET STREQUAL "plain-mpi")
      set(MPIEXEC_NUMCORE_FLAG "" CACHE STRING "${MPIEXEC_NUMCORE_FLAG_DESCRIPTION}" FORCE)
      set(MPIEXEC_NUMCORES_PER_RANK "" CACHE STRING "${MPIEXEC_NUMCORES_PER_RANK_DESCRIPTION}" FORCE)
      if(DLAF_TEST_THREAD_BINDING_ENABLED)
        message(WARNING "Disabling pika binding")
        set(DLAF_TEST_THREAD_BINDING_ENABLED FALSE CACHE BOOL "" FORCE)
      endif()

    elseif(DLAF_MPI_PRESET STREQUAL "slurm")
      if(NOT DLAF_TEST_THREAD_BINDING_ENABLED)
        message(
          WARNING "When using DLAF_MPI_PRESET=slurm DLAF_TEST_THREAD_BINDING_ENABLED should be enabled. "
                  "It is currently disabled and you may incur in a performance drop. "
                  "Leave it disabled at your own risk."
        )
      endif()

      execute_process(
        COMMAND which srun OUTPUT_VARIABLE SLURM_EXECUTABLE OUTPUT_STRIP_TRAILING_WHITESPACE
      )
      set(MPIEXEC_EXECUTABLE ${SLURM_EXECUTABLE} CACHE STRING "${MPIEXEC_EXECUTABLE_DESCRIPTION}" FORCE)
      set(MPIEXEC_NUMPROC_FLAG "-n" CACHE STRING "${MPIEXEC_NUMPROC_FLAG_DESCRIPTION}" FORCE)
      set(MPIEXEC_NUMCORE_FLAG "-c" CACHE STRING "${MPIEXEC_NUMCORE_FLAG_DESCRIPTION}" FORCE)
      set(MPIEXEC_NUMCORES_PER_RANK "1" CACHE STRING "${MPIEXEC_NUMCORES_PER_RANK_DESCRIPTION}")

    elseif(DLAF_MPI_PRESET STREQUAL "custom")
      # nothing to do here
    else()
      message(FATAL_ERROR "Preset ${DLAF_MPI_PRESET} is not supported")
    endif()

    message(STATUS "MPI preset: ${DLAF_MPI_PRESET}")

    # make mpi preset selection persistent (with the aim to not overwrite each time, user may have changed some values (see custom)
    set(_DLAF_MPI_PRESET ${DLAF_MPI_PRESET} CACHE INTERNAL "Store what MPI preset is being used")

    mark_as_advanced(
      MPIEXEC_EXECUTABLE MPIEXEC_NUMPROC_FLAG MPIEXEC_NUMCORE_FLAG MPIEXEC_NUMCORES_PER_RANK
    )
  endif()

  # ----- MPI
  # there must be an mpiexec, otherwise it will not be possible to run MPI based tests
  if(NOT MPIEXEC_EXECUTABLE)
    message(FATAL_ERROR "Please set MPIEXEC_EXECUTABLE to run MPI tests.")
  endif()

  # if a numcore flag is specified, it must be specified also the number of cores per node (and viceversa)
  if((MPIEXEC_NUMCORE_FLAG OR MPIEXEC_NUMCORES_PER_RANK) AND NOT (MPIEXEC_NUMCORE_FLAG
                                                                  AND MPIEXEC_NUMCORES_PER_RANK)
  )
    message(WARNING "MPIEXEC_NUMCORES_PER_RANK = '${MPIEXEC_NUMCORES_PER_RANK}'")
    message(WARNING "MPIEXEC_NUMCORE_FLAG = '${MPIEXEC_NUMCORE_FLAG}'")
    message(
      FATAL_ERROR
        "MPIEXEC_NUMCORES_PER_RANK and MPIEXEC_NUMCORE_FLAG must be either both set or both empty."
    )
  endif()
endmacro()

# Check if LIST_NAME contains at least an element that matches ELEMENT_REGEX. If not, add FALLBACK
# to the list.
function(_set_element_to_fallback_value LIST_NAME ELEMENT_REGEX FALLBACK)
  set(_TMP_LIST ${${LIST_NAME}})
  list(FILTER _TMP_LIST INCLUDE REGEX ${ELEMENT_REGEX})
  list(LENGTH _TMP_LIST _NUM_TMP_LIST)
  if(_NUM_TMP_LIST EQUAL 0)
    list(APPEND ${LIST_NAME} ${FALLBACK})
    set(${LIST_NAME} ${${LIST_NAME}} PARENT_SCOPE)
  endif()
endfunction()

# DLAF_addTargetTest(test_target_name
#   [ARGUMENTS <command-line-arguments-for-test-executable>]
#   [MPIRANKS <number of rank>]
#   [USE_MAIN {PLAIN | PIKA | MPI | MPIPIKA}]
#   [CATEGORY <category>]
# )
#
# MPIRANKS specifies the number of ranks on which the test will be carried out and it implies a link with
# MPI library. At build time the constant NUM_MPI_RANKS=MPIRANKS is set.
#
# USE_MAIN links to an external main function, in particular:
#   - PLAIN: uses the classic gtest_main
#   - PIKA: uses a main that initializes pika
#   - MPI: uses a main that initializes MPI
#   - MPIPIKA: uses a main that initializes both pika and MPI
# If not specified, no external main is used and it should exist in the test source code.
#
# Moreover, there are a few variables to control the behavior:
# - DLAF_PIKATEST_EXTRA_ARGS can be used to pass extra arguments that will be given to all tests involving PIKA (i.e. USE_MAIN=PIKA or USE_MAIN=MPIPIKA).
# - MPIEXEC_MAX_NUMPROCS is the maximum number of ranks that could be run
# - MPIEXEC_NUMCORE_FLAG can be set to the MPI runner flag that controls the number of cores per rank (see MPIEXEC_NUMCORES_PER_RANK).
# - MPIEXEC_NUMCORES_PER_RANK can be set to number of cores to assign to each MPI rank (default=1)
#
# The usage of the aforementioned MPIEXEC_* variables depends on the DLAF_MPI_PRESET value:
# - With {"plain-mpi"}
#   MPIEXEC_MAX_NUMPROCS is considered, and the number of processors is distributed evenly over MPIRANKS (as --pika:threads)
# - With {"slurm", "custom"}
#   MPIEXEC_NUMCORES_PER_RANK is used to set the number of cores for each rank.
#   It can be set only if MPIEXEC_NUMCORE_FLAG is set, otherwise the information is ignored.
#
# e.g.
#
# DLAF_addTargetTest(
#   example_test
#   USE_MAIN
#   MPIPIKA
#   MPIRANKS 6
#   ARGUMENTS
#     --grid-rows=3
#     --grid-cols=2
#     --check=all
#   CATEGORY MINIAPP
# )
function(DLAF_addTargetTest test_target_name)
  set(options "")
  set(oneValueArgs CATEGORY MPIRANKS USE_MAIN)
  set(multiValueArgs ARGUMENTS)
  cmake_parse_arguments(DLAF_ATT "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  ### Checks
  if(DLAF_ATT_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "Unknown arguments ${DLAF_ATT_UNPARSED_ARGUMENTS}")
  endif()

  set(IS_AN_MPI_TEST FALSE)
  set(IS_AN_PIKA_TEST FALSE)
  if(NOT DLAF_ATT_USE_MAIN OR DLAF_ATT_USE_MAIN STREQUAL PLAIN)

  elseif(DLAF_ATT_USE_MAIN STREQUAL PIKA)
    set(IS_AN_PIKA_TEST TRUE)
  elseif(DLAF_ATT_USE_MAIN STREQUAL MPI)
    set(IS_AN_MPI_TEST TRUE)
  elseif(DLAF_ATT_USE_MAIN STREQUAL MPIPIKA)
    set(IS_AN_MPI_TEST TRUE)
    set(IS_AN_PIKA_TEST TRUE)
  elseif(DLAF_ATT_USE_MAIN STREQUAL CAPI)
    set(IS_AN_MPI_TEST TRUE)
  else()
    message(FATAL_ERROR "USE_MAIN=${DLAF_ATT_USE_MAIN} is not a supported option")
  endif()

  set(_TEST_LABELS)

  if(NOT DLAF_ATT_CATEGORY)
    set(DLAF_ATT_CATEGORY "UNIT")
  endif()

  list(APPEND _TEST_LABELS "CATEGORY_${DLAF_ATT_CATEGORY}")

  if(IS_AN_MPI_TEST)
    if(NOT DLAF_ATT_MPIRANKS)
      message(FATAL_ERROR "You are asking for an MPI external main without specifying MPIRANKS")
    endif()
    if(NOT DLAF_ATT_MPIRANKS GREATER 0)
      message(FATAL_ERROR "Wrong MPIRANKS number ${DLAF_ATT_MPIRANKS}")
    endif()
    if(DLAF_MPI_PRESET STREQUAL "plain-mpi" AND (DLAF_ATT_MPIRANKS GREATER MPIEXEC_MAX_NUMPROCS))
      message(
        WARNING
          "\
      YOU ARE ASKING FOR ${DLAF_ATT_MPIRANKS} RANKS, BUT THERE ARE JUST ${MPIEXEC_MAX_NUMPROCS} CORES.
      You can adjust MPIEXEC_MAX_NUMPROCS value to suppress this warning.
      Using OpenMPI may require to set the environment variable OMPI_MCA_rmaps_base_oversubscribe=1."
      )
    endif()
  else()
    if(DLAF_ATT_MPIRANKS)
      message(FATAL_ERROR "You specified MPIRANKS and asked for an external main without MPI")
    else()
      set(DLAF_ATT_MPIRANKS 1)
    endif()
  endif()

  ### Test target
  set(DLAF_TEST_RUNALL_WITH_MPIEXEC OFF CACHE BOOL "Run all tests using the workload manager.")

  set(_TEST_ARGUMENTS ${DLAF_ATT_ARGUMENTS})

  if(DLAF_TEST_RUNALL_WITH_MPIEXEC OR IS_AN_MPI_TEST)
    if(MPIEXEC_NUMCORE_FLAG)
      if(MPIEXEC_NUMCORES_PER_RANK)
        set(DLAF_CORES_PER_RANK ${MPIEXEC_NUMCORES_PER_RANK})
      else()
        set(DLAF_CORES_PER_RANK 1)
      endif()

      set(_MPI_CORE_ARGS ${MPIEXEC_NUMCORE_FLAG} ${DLAF_CORES_PER_RANK})
    else()
      set(_MPI_CORE_ARGS "")
    endif()

    if(DLAF_CI_RUNNER_USES_MPIRUN)
      set(_TEST_COMMAND ${DLAF_TEST_PREFLAGS} $<TARGET_FILE:${test_target_name}> ${DLAF_TEST_POSTFLAGS})
    else()
      separate_arguments(MPIEXEC_PREFLAGS)
      set(_TEST_COMMAND
          ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${DLAF_ATT_MPIRANKS} ${_MPI_CORE_ARGS}
          ${MPIEXEC_PREFLAGS} ${DLAF_TEST_PREFLAGS} $<TARGET_FILE:${test_target_name}>
          ${DLAF_TEST_POSTFLAGS} ${MPIEXEC_POSTFLAGS}
      )
    endif()
    list(APPEND _TEST_LABELS "RANK_${DLAF_ATT_MPIRANKS}")
  else()
    # ----- Classic test
    set(_TEST_COMMAND ${DLAF_TEST_PREFLAGS} $<TARGET_FILE:${test_target_name}> ${DLAF_TEST_POSTFLAGS})
    list(APPEND _TEST_LABELS "RANK_1")
  endif()

  if(IS_AN_PIKA_TEST)
    separate_arguments(_PIKA_EXTRA_ARGS_LIST UNIX_COMMAND ${DLAF_PIKATEST_EXTRA_ARGS})

    # --pika:bind=none is useful just in case more ranks are going to be allocated on the same node.
    if(IS_AN_MPI_TEST AND (DLAF_ATT_MPIRANKS GREATER 1) AND (NOT DLAF_TEST_THREAD_BINDING_ENABLED))
      _set_element_to_fallback_value(_PIKA_EXTRA_ARGS_LIST "--pika:bind" "--pika:bind=none")
    endif()

    if(IS_AN_MPI_TEST AND DLAF_MPI_PRESET STREQUAL "plain-mpi")
      math(EXPR _DLAF_PIKA_THREADS "${MPIEXEC_MAX_NUMPROCS}/${DLAF_ATT_MPIRANKS}")

      if(_DLAF_PIKA_THREADS LESS 2)
        set(_DLAF_PIKA_THREADS 2)
      endif()

      _set_element_to_fallback_value(
        _PIKA_EXTRA_ARGS_LIST "--pika:threads" "--pika:threads=${_DLAF_PIKA_THREADS}"
      )
    endif()

    list(APPEND _TEST_ARGUMENTS ${_PIKA_EXTRA_ARGS_LIST})
  endif()

  # Special treatment for C API tests
  # C API tests require pika arguments to be hard-coded in the test file
  if(DLAF_ATT_USE_MAIN STREQUAL CAPI)
    separate_arguments(_PIKA_EXTRA_ARGS_LIST_CAPI UNIX_COMMAND ${DLAF_PIKATEST_EXTRA_ARGS})

    # --pika:bind=none is useful just in case more ranks are going to be allocated on the same node.
    if((DLAF_ATT_MPIRANKS GREATER 1) AND (NOT DLAF_TEST_THREAD_BINDING_ENABLED))
      _set_element_to_fallback_value(_PIKA_EXTRA_ARGS_LIST_CAPI "--pika:bind" "--pika:bind=none")
    endif()

    if(IS_AN_MPI_TEST AND DLAF_MPI_PRESET STREQUAL "plain-mpi")
      math(EXPR _DLAF_PIKA_THREADS "${MPIEXEC_MAX_NUMPROCS}/${DLAF_ATT_MPIRANKS}")

      if(_DLAF_PIKA_THREADS LESS 2)
        set(_DLAF_PIKA_THREADS 2)
      endif()

      _set_element_to_fallback_value(
        _PIKA_EXTRA_ARGS_LIST_CAPI "--pika:threads" "--pika:threads=${_DLAF_PIKA_THREADS}"
      )
    endif()

    string(REPLACE ";" "\", \"" PIKA_EXTRA_ARGS_LIST_CAPI "${_PIKA_EXTRA_ARGS_LIST_CAPI}")

    configure_file(
      ${PROJECT_SOURCE_DIR}/test/include/dlaf_c_test/config.h.in
      ${CMAKE_CURRENT_BINARY_DIR}/${test_target_name}_config.h
    )

  endif()

  add_test(NAME ${test_target_name} COMMAND ${_TEST_COMMAND} ${_TEST_ARGUMENTS})
  set_tests_properties(${test_target_name} PROPERTIES LABELS "${_TEST_LABELS}")
endfunction()

# DLAF_addTest(test_target_name
#   SOURCES <source1> [<source2> ...]
#   [COMPILE_DEFINITIONS <arguments for target_compile_definitions>]
#   [INCLUDE_DIRS <arguments for target_include_directories>]
#   [LIBRARIES <arguments for target_link_libraries>]
#   [ARGUMENTS <command-line-arguments-for-test-executable>]
#   [MPIRANKS <number of rank>]
#   [USE_MAIN {PLAIN | PIKA | MPI | MPIPIKA}]
#   [CATEGORY <category>]
# )
#
# Utility function over DLAF_addTargetTest that takes care of creating the CMake target.
#
# At least one source file has to be specified, while other parameters are optional.
#
# COMPILE_DEFINITIONS, INCLUDE_DIRS and LIBRARIES are passed to respective cmake wrappers, so it is
# possible to specify PRIVATE/INTERFACE/PUBLIC modifiers.
#
# For all other options see DLAF_addTargetTest documentation.
#
# e.g.
# DLAF_addTest(example_test
#   SOURCE main.cpp testfixture.cpp
#   LIBRARIES
#     PRIVATE
#       boost::boost
#       include/
# )
function(DLAF_addTest test_target_name)
  set(options "")
  set(oneValueArgs CATEGORY MPIRANKS USE_MAIN)
  set(multiValueArgs SOURCES COMPILE_DEFINITIONS INCLUDE_DIRS LIBRARIES ARGUMENTS)
  cmake_parse_arguments(DLAF_AT "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  ### Checks
  if(DLAF_AT_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "Unknown arguments ${DLAF_AT_UNPARSED_ARGUMENTS}")
  endif()

  if(NOT DLAF_AT_SOURCES)
    message(FATAL_ERROR "No sources specified for this test")
  endif()

  if(NOT DLAF_AT_USE_MAIN)
    set(_gtest_tgt gtest)
  elseif(DLAF_AT_USE_MAIN STREQUAL PLAIN)
    set(_gtest_tgt gtest_main)
  elseif(DLAF_AT_USE_MAIN STREQUAL PIKA)
    set(_gtest_tgt DLAF_gtest_pika_main)
  elseif(DLAF_AT_USE_MAIN STREQUAL MPI)
    set(_gtest_tgt DLAF_gtest_mpi_main)
  elseif(DLAF_AT_USE_MAIN STREQUAL MPIPIKA)
    set(_gtest_tgt DLAF_gtest_mpipika_main)
  elseif(DLAF_AT_USE_MAIN STREQUAL CAPI)
    set(_gtest_tgt DLAF_gtest_mpi_main)
  else()
    message(FATAL_ERROR "USE_MAIN=${DLAF_AT_USE_MAIN} is not a supported option")
  endif()

  if(NOT DLAF_AT_CATEGORY)
    set(DLAF_AT_CATEGORY "UNIT")
  endif()

  ### Test executable target
  add_executable(${test_target_name} ${DLAF_AT_SOURCES})
  target_link_libraries(
    ${test_target_name} PRIVATE ${_gtest_tgt} DLAF_test ${DLAF_AT_LIBRARIES} dlaf.prop_private
  )
  set(IS_AN_MPI_TEST FALSE)
  if(DLAF_AT_USE_MAIN MATCHES MPI OR DLAF_AT_USE_MAIN STREQUAL CAPI)
    set(IS_AN_MPI_TEST TRUE)
  endif()
  target_compile_definitions(
    ${test_target_name} PRIVATE ${DLAF_AT_COMPILE_DEFINITIONS} $<$<BOOL:${IS_AN_MPI_TEST}>:
                                NUM_MPI_RANKS=${DLAF_AT_MPIRANKS}>
  )
  target_include_directories(
    ${test_target_name} PRIVATE ${DLAF_AT_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR}
  )
  target_add_warnings(${test_target_name})
  DLAF_addPrecompiledHeaders(${test_target_name})

  ### DEPLOY
  include(GNUInstallDirs)

  set(DLAF_INSTALL_TESTS OFF CACHE BOOL "If tests are built, it controls if they will be installed")
  if(DLAF_INSTALL_TESTS)
    install(TARGETS ${test_target_name}
                    # EXPORT DLAF-tests
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
  endif()

  ### Test
  DLAF_addTargetTest(
    ${test_target_name}
    MPIRANKS ${DLAF_AT_MPIRANKS}
    USE_MAIN ${DLAF_AT_USE_MAIN}
    ARGUMENTS ${DLAF_AT_ARGUMENTS}
    CATEGORY ${DLAF_AT_CATEGORY}
  )
endfunction()
