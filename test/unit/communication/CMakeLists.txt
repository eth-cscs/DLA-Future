#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2021, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

DLAF_addTest(test_message
  SOURCES test_message.cpp
  LIBRARIES dlaf.core
  MPIRANKS 2
  USE_MAIN MPI
)

DLAF_addTest(test_communicator
  SOURCES test_communicator.cpp
  LIBRARIES dlaf.core
  MPIRANKS 4
  USE_MAIN MPI
)

DLAF_addTest(test_communicator_grid
  SOURCES test_communicator_grid.cpp
  LIBRARIES dlaf.core
  MPIRANKS 6
  USE_MAIN MPI
)

DLAF_addTest(test_broadcast
  SOURCES test_broadcast.cpp
  LIBRARIES dlaf.core
  MPIRANKS 4
  USE_MAIN MPI
)

DLAF_addTest(test_reduce
  SOURCES test_reduce.cpp
  LIBRARIES dlaf.core
  MPIRANKS 4
  USE_MAIN MPI
)

DLAF_addTest(test_all_reduce
  SOURCES test_all_reduce.cpp
  LIBRARIES dlaf.core
  MPIRANKS 4
  USE_MAIN MPI
)

DLAF_addTest(test_broadcast_tile
  SOURCES test_broadcast_tile.cpp
  LIBRARIES dlaf.core
  MPIRANKS 4
  USE_MAIN MPI
)

DLAF_addTest(test_broadcast_matrix
  SOURCES test_broadcast_matrix.cpp
  LIBRARIES dlaf.core
  MPIRANKS 4
  USE_MAIN MPIHPX
)

DLAF_addTest(test_broadcast_panel
  SOURCES test_broadcast_panel.cpp
  LIBRARIES dlaf.core
  MPIRANKS 6
  USE_MAIN MPIHPX
)

DLAF_addTest(test_comm_executor
  SOURCES test_comm_executor.cpp
  LIBRARIES dlaf.core
  MPIRANKS 4
  USE_MAIN MPIHPX
)

DLAF_addTest(test_comm_matrix
  SOURCES test_comm_matrix.cpp
  LIBRARIES dlaf.core
  MPIRANKS 2
  USE_MAIN MPIHPX
)

DLAF_addTest(test_collective_async
  SOURCES test_collective_async.cpp
  LIBRARIES dlaf.core
  MPIRANKS 2
  USE_MAIN MPIHPX
)
