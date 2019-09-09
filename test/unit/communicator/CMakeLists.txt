DLAF_addTest(test_communicator
  SOURCES test_communicator.cpp
  MPIRANKS 4
  USE_GTEST_MAIN
)

DLAF_addTest(test_communicator_grid
  SOURCES test_communicator_grid.cpp
  MPIRANKS 4
  USE_GTEST_MAIN
)
