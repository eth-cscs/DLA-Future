#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2023, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

# DLAF_addMiniapp(miniapp_target_name
#   SOURCES <source1> [<source2> ...]
#   [COMPILE_DEFINITIONS <arguments for target_compile_definitions>]
#   [INCLUDE_DIRS <arguments for target_include_directories>]
#   [LIBRARIES <arguments for target_link_libraries>]
# )
#
# At least one source file has to be specified, while other parameters are optional.
#
# COMPILE_DEFINITIONS, INCLUDE_DIRS and LIBRARIES are passed to respective cmake wrappers, so it is
# possible to specify PRIVATE/INTERFACE/PUBLIC modifiers.
#
# e.g.
#
# DLAF_addMiniapp(example_miniapp
#   SOURCE main.cpp
#   LIBRARIES
#     PRIVATE
#       boost::boost
#       include/
# )

function(DLAF_addMiniapp miniapp_target_name)
  set(options TEST)
  set(oneValueArgs MPIRANKS USE_MAIN)
  set(multiValueArgs SOURCES COMPILE_DEFINITIONS INCLUDE_DIRS LIBRARIES ARGUMENTS)
  cmake_parse_arguments(DLAF_AM "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  ### Checks
  if(DLAF_AM_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "Unknown arguments ${DLAF_AM_UNPARSED_ARGUMENTS}")
  endif()

  if(NOT DLAF_AM_SOURCES)
    message(FATAL_ERROR "No sources specified for this miniapp")
  endif()

  ### Miniapp executable target
  add_executable(${miniapp_target_name} ${DLAF_AM_SOURCES})
  target_link_libraries(
    ${miniapp_target_name} PRIVATE DLAF_miniapp ${DLAF_AM_LIBRARIES} DLAF::dlaf.prop_private
  )
  target_compile_definitions(${miniapp_target_name} PRIVATE ${DLAF_AM_COMPILE_DEFINITIONS})
  target_include_directories(${miniapp_target_name} PRIVATE ${DLAF_AM_INCLUDE_DIRS})
  target_add_warnings(${miniapp_target_name})
  ### DEPLOY
  include(GNUInstallDirs)

  set(DLAF_INSTALL_MINIAPPS ON CACHE BOOL "If miniapps are built, it controls if they will be installed")
  if(DLAF_INSTALL_MINIAPPS)
    install(TARGETS ${miniapp_target_name}
                    # EXPORT DLAF-miniapps
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
  endif()

  if(DLAF_AM_TEST)
    # TODO: Deduplicate this from DLAF_AddTest.cmake
    set(_TEST_ARGUMENTS)
    set(DLAF_AT_MPIRANKS 4)
  
    if(MPIEXEC_NUMCORE_FLAG)
      if(MPIEXEC_NUMCORES)
        set(_CORES_PER_RANK ${MPIEXEC_NUMCORES})
      else()
        set(_CORES_PER_RANK 1)
      endif()
  
      math(EXPR DLAF_CORE_PER_RANK "${_CORES_PER_RANK}/${DLAF_AT_MPIRANKS}")
  
      if(NOT DLAF_CORE_PER_RANK)
        set(DLAF_CORE_PER_RANK 1)
      endif()
  
      set(_MPI_CORE_ARGS ${MPIEXEC_NUMCORE_FLAG} ${DLAF_CORE_PER_RANK})
    else()
      set(_MPI_CORE_ARGS "")
    endif()
  
    if(DLAF_CI_RUNNER_USES_MPIRUN)
      separate_arguments(DLAF_TEST_PRECOMMAND)
      set(_TEST_COMMAND ${DLAF_TEST_PRECOMMAND} $<TARGET_FILE:${miniapp_target_name}>)
    else()
      separate_arguments(MPIEXEC_PREFLAGS)
      separate_arguments(DLAF_TEST_PRECOMMAND)
      set(_TEST_COMMAND
          ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${DLAF_AT_MPIRANKS} ${_MPI_CORE_ARGS}
          ${MPIEXEC_PREFLAGS} ${DLAF_TEST_PRECOMMAND} $<TARGET_FILE:${miniapp_target_name}> ${MPIEXEC_POSTFLAGS}
      )
    endif()
    set(_TEST_LABEL "RANK_${DLAF_AT_MPIRANKS}")
  
    separate_arguments(_PIKA_EXTRA_ARGS_LIST UNIX_COMMAND ${DLAF_PIKATEST_EXTRA_ARGS})
  
    # --pika:bind=none is useful just in case more ranks are going to be allocated on the same node.
    if(IS_AN_MPI_TEST AND (DLAF_AT_MPIRANKS GREATER 1) AND (NOT DLAF_TEST_THREAD_BINDING_ENABLED))
      _set_element_to_fallback_value(_PIKA_EXTRA_ARGS_LIST "--pika:bind" "--pika:bind=none")
    endif()
  
    if(IS_AN_MPI_TEST AND DLAF_MPI_PRESET STREQUAL "plain-mpi")
      math(EXPR _DLAF_PIKA_THREADS "${MPIEXEC_MAX_NUMPROCS}/${DLAF_AT_MPIRANKS}")
  
      if(_DLAF_PIKA_THREADS LESS 2)
        set(_DLAF_PIKA_THREADS 2)
      endif()
  
      _set_element_to_fallback_value(
        _PIKA_EXTRA_ARGS_LIST "--pika:threads" "--pika:threads=${_DLAF_PIKA_THREADS}"
      )
    endif()
  
    list(APPEND _TEST_ARGUMENTS ${_PIKA_EXTRA_ARGS_LIST})

    add_test(NAME ${miniapp_target_name} COMMAND ${_TEST_COMMAND} ${_TEST_ARGUMENTS} --grid-rows=2 --grid-cols=2)
    set_tests_properties(${miniapp_target_name} PROPERTIES LABELS "RANK_4;MINIAPP")
  endif()
endfunction()
