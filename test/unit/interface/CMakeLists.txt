#
#Distributed Linear Algebra with Future(DLAF)
#
#Copyright(c) 2018 - 2023, ETH Zurich
#All rights reserved.
#
#Please, refer to the LICENSE file in the root directory.
#SPDX - License - Identifier : BSD - 3 - Clause
#

if (DLAF_WITH_INTERFACE)
  DLAF_addTest(test_cholesky_interface
    SOURCES test_cholesky.cpp
    LIBRARIES dlaf.interface dlaf.core
    USE_MAIN MPI MPIRANKS 6
  )

  # DLAF_addTest(test_eigensolver_interface
  #   SOURCES test_eigensolver.cpp
  #   LIBRARIES dlaf.interface dlaf.core
  #   USE_MAIN MPI MPIRANKS 6
  # )

endif()
