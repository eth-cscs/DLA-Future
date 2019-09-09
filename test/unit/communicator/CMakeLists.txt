add_executable(test_communicator test_communicator.cpp)
target_link_libraries(test_communicator
  PRIVATE
    DLAF
    MPI::MPI_CXX
)

add_executable(test_communicator_grid test_communicator_grid.cpp)
target_link_libraries(test_communicator_grid
  PRIVATE
    DLAF
    MPI::MPI_CXX
)
