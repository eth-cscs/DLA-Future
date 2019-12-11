#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2019, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

set(DLAF_PRESET_OPTIONS "plain-mpi" "slurm" "custom")
set(DLAF_MPI_PRESET "plain-mpi" CACHE STRING "Select a preset to use")
set_property(CACHE DLAF_MPI_PRESET PROPERTY STRINGS ${DLAF_PRESET_OPTIONS})

# if a preset has been selected and it has been changed from previous configurations
if (NOT DLAF_MPI_PRESET STREQUAL _DLAF_MPI_PRESET)

  if(DLAF_MPI_PRESET STREQUAL "plain-mpi")
    # set(MPIEXEC_EXECUTABLE "" CACHE STRING "Executable for running MPI programs")
    # set(MPIEXEC_NUMPROC_FLAG "" CACHE STRING "Flag used by MPI to specify the number of processes for mpiexec; the next option will be the number of processes." FORCE)
    set(MPIEXEC_NUMCORE_FLAG "" CACHE STRING "Flag used by MPI to specify the number of cores per rank for mpiexec. If not empty, you have to specify also the number of cores available per node in MPIEXEC_NUMCORES." FORCE)
    set(MPIEXEC_NUMCORES "" CACHE STRING "Number of cores available for each MPI rank." FORCE)

  elseif(DLAF_MPI_PRESET STREQUAL "slurm")
    execute_process(COMMAND which srun OUTPUT_VARIABLE SLURM_EXECUTABLE OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(MPIEXEC_EXECUTABLE ${SLURM_EXECUTABLE} CACHE STRING "Executable for running MPI programs" FORCE)
    set(MPIEXEC_NUMPROC_FLAG "-n" CACHE STRING "Flag used by MPI to specify the number of processes for mpiexec; the next option will be the number of processes." FORCE)
    set(MPIEXEC_NUMCORE_FLAG "-c" CACHE STRING "Flag used by MPI to specify the number of cores per rank for mpiexec. If not empty, you have to specify also the number of cores available per node in MPIEXEC_NUMCORES." FORCE)
    set(MPIEXEC_NUMCORES "1" CACHE STRING "Number of cores available for each MPI rank.")

  elseif(DLAF_MPI_PRESET STREQUAL "custom")
    set(MPIEXEC_EXECUTABLE "" CACHE STRING "Executable for running MPI programs")
    set(MPIEXEC_NUMPROC_FLAG "" CACHE STRING "Flag used by MPI to specify the number of processes for mpiexec; the next option will be the number of processes.")
    set(MPIEXEC_NUMCORE_FLAG "" CACHE STRING "Flag used by MPI to specify the number of cores per rank for mpiexec. If not empty, you have to specify also the number of cores available per node in MPIEXEC_NUMCORES.")
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
if (NOT MPIEXEC_EXECUTABLE)
  message(FATAL_ERROR "Please set MPIEXEC_EXECUTABLE to run MPI tests.")
endif()

# if a numcore flag is specified, it must be specified also the number of cores per node (and viceversa)
if ((MPIEXEC_NUMCORE_FLAG OR MPIEXEC_NUMCORES) AND NOT (MPIEXEC_NUMCORE_FLAG AND MPIEXEC_NUMCORES))
  message(FATAL_ERROR "MPIEXEC_NUMCORES and MPIEXEC_NUMCORE_FLAG must be either both sets or both empty.")
endif()

# ----- HPX
set(DLAF_HPXTEST_EXTRA_ARGS "" CACHE STRING "Extra arguments for tests with HPX")


add_library(DLAF_test src/util_mpi.cpp)

target_include_directories(DLAF_test
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(DLAF_test
  PRIVATE
    MPI::MPI_CXX
)

add_subdirectory(src)
add_subdirectory(unit)
