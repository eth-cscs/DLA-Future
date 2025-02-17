#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

macro(dlaf_setup_mpi_preset)
  # if a preset has been selected and it has been changed from previous configurations
  if(NOT DLAF_MPI_PRESET STREQUAL _DLAF_MPI_PRESET)

    if(DLAF_MPI_PRESET STREQUAL "plain-mpi")
      # set(MPIEXEC_EXECUTABLE "" CACHE STRING "Executable for running MPI programs")
      # set(MPIEXEC_NUMPROC_FLAG "" CACHE STRING "Flag used by MPI to specify the number of processes for mpiexec; the next option will be the number of processes." FORCE)
      set(MPIEXEC_NUMCORE_FLAG
          ""
          CACHE
            STRING
            "Flag used by MPI to specify the number of cores per rank for mpiexec. If not empty, you have to specify also the number of cores available per node in MPIEXEC_NUMCORES."
            FORCE
      )
      set(MPIEXEC_NUMCORES "" CACHE STRING "Number of cores available for each MPI rank." FORCE)
      if(DLAF_TEST_THREAD_BINDING_ENABLED)
        message(WARNING "Disabling pika binding")
        set(DLAF_TEST_THREAD_BINDING_ENABLED FALSE CACHE BOOL "" FORCE)
      endif()

    elseif(DLAF_MPI_PRESET STREQUAL "slurm")
      if(NOT DLAF_TEST_THREAD_BINDING_ENABLED)
        message(
          WARNING "When using DLAF_MPI_PRESET=slurm DLAF_TEST_THREAD_BINDING_ENABLED should be enabled. "
                  "It is currently disabled and you may incur in performance drop. "
                  "Leave it disabled at your own risk."
        )
      endif()

      execute_process(
        COMMAND which srun OUTPUT_VARIABLE SLURM_EXECUTABLE OUTPUT_STRIP_TRAILING_WHITESPACE
      )
      set(MPIEXEC_EXECUTABLE ${SLURM_EXECUTABLE} CACHE STRING "Executable for running MPI programs"
                                                       FORCE
      )
      set(MPIEXEC_NUMPROC_FLAG
          "-n"
          CACHE
            STRING
            "Flag used by MPI to specify the number of processes for mpiexec; the next option will be the number of processes."
            FORCE
      )
      set(MPIEXEC_NUMCORE_FLAG
          "-c"
          CACHE
            STRING
            "Flag used by MPI to specify the number of cores per rank for mpiexec. If not empty, you have to specify also the number of cores available per node in MPIEXEC_NUMCORES."
            FORCE
      )
      set(MPIEXEC_NUMCORES "1" CACHE STRING "Number of cores available for each MPI rank.")

    elseif(DLAF_MPI_PRESET STREQUAL "custom")
      set(MPIEXEC_EXECUTABLE "" CACHE STRING "Executable for running MPI programs")
      set(MPIEXEC_NUMPROC_FLAG
          ""
          CACHE
            STRING
            "Flag used by MPI to specify the number of processes for mpiexec; the next option will be the number of processes."
      )
      set(MPIEXEC_NUMCORE_FLAG
          ""
          CACHE
            STRING
            "Flag used by MPI to specify the number of cores per rank for mpiexec. If not empty, you have to specify also the number of cores available per node in MPIEXEC_NUMCORES."
      )
      set(MPIEXEC_NUMCORES "" CACHE STRING "Number of cores available for each MPI rank.")

    else()
      message(FATAL_ERROR "Preset ${DLAF_MPI_PRESET} is not supported")
    endif()

    message(STATUS "MPI preset: ${DLAF_MPI_PRESET}")

    # make mpi preset selection persistent (with the aim to not overwrite each time, user may have changed some values (see custom)
    set(_DLAF_MPI_PRESET ${DLAF_MPI_PRESET} CACHE INTERNAL "Store what preset is being used")

    mark_as_advanced(MPIEXEC_EXECUTABLE MPIEXEC_NUMPROC_FLAG MPIEXEC_NUMCORE_FLAG MPIEXEC_NUMCORES)
  endif()

  # ----- MPI
  # there must be an mpiexec, otherwise it will not be possible to run MPI based tests
  if(NOT MPIEXEC_EXECUTABLE)
    message(FATAL_ERROR "Please set MPIEXEC_EXECUTABLE to run MPI tests.")
  endif()

  # if a numcore flag is specified, it must be specified also the number of cores per node (and viceversa)
  if((MPIEXEC_NUMCORE_FLAG OR MPIEXEC_NUMCORES) AND NOT (MPIEXEC_NUMCORE_FLAG AND MPIEXEC_NUMCORES))
    message(
      FATAL_ERROR "MPIEXEC_NUMCORES and MPIEXEC_NUMCORE_FLAG must be either both sets or both empty."
    )
  endif()
endmacro()

# Loads all scripts available in CMAKE_MODULE_PATH with filename "DLAF_*.cmake"
foreach(module_folder IN LISTS CMAKE_MODULE_PATH)
  file(GLOB cmake_modules "${module_folder}/DLAF_*.cmake")

  foreach(script_filepath IN LISTS cmake_modules)
    get_filename_component(script_name ${script_filepath} NAME_WE)
    include(${script_name})
  endforeach()
endforeach()
