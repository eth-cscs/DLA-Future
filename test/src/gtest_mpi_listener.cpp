// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "gtest_mpi_listener.h"

#include <algorithm>

#define MASTER_CALLS_DEFAULT_LISTENER(name, ...) \
  if (isMasterRank())                            \
    listener_->name(__VA_ARGS__);

namespace internal {
void mpi_send_string(const std::string& message, int to_rank);
std::string mpi_receive_string(int from_rank);
}

MPIListener::MPIListener(int argc, char** argv, ::testing::TestEventListener* other)
    : listener_(std::move(other)) {
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
}

void MPIListener::OnTestProgramStart(const ::testing::UnitTest& unit_test) {
  MASTER_CALLS_DEFAULT_LISTENER(OnTestProgramStart, unit_test);
}

void MPIListener::OnTestIterationStart(const ::testing::UnitTest& unit_test, int iteration) {
  MASTER_CALLS_DEFAULT_LISTENER(OnTestIterationStart, unit_test, iteration);
}

void MPIListener::OnEnvironmentsSetUpStart(const ::testing::UnitTest& unit_test) {
  MASTER_CALLS_DEFAULT_LISTENER(OnEnvironmentsSetUpStart, unit_test);
}

void MPIListener::OnEnvironmentsSetUpEnd(const ::testing::UnitTest& unit_test) {
  MASTER_CALLS_DEFAULT_LISTENER(OnEnvironmentsSetUpEnd, unit_test);
}

void MPIListener::OnTestCaseStart(const ::testing::TestCase& test_case) {
  MASTER_CALLS_DEFAULT_LISTENER(OnTestCaseStart, test_case);
}

void MPIListener::OnTestStart(const ::testing::TestInfo& test_info) {
  MASTER_CALLS_DEFAULT_LISTENER(OnTestStart, test_info);
}

void MPIListener::OnTestPartResult(const ::testing::TestPartResult& test_part_result) {
  std::ostringstream os;
  os << test_part_result;
  last_test_part_results_.push_back(os.str());
}

void MPIListener::OnTestEnd(const ::testing::TestInfo& test_info) {
  auto print_partial_results = [this](int rank, int total_results,
                                      std::function<std::string(int)> get_result) {
    if (total_results <= 0)
      return;

    printf("[ RANK  %2d ]\n", rank);
    for (auto index_result = 0; index_result < total_results; ++index_result)
      printf("%s", get_result(index_result).c_str());
  };

  for (int rank = 0; rank < world_size_; ++rank) {
    if (isMasterRank()) {
      int number_of_results;
      std::function<std::string(int)> get_result;

      if (rank == 0) {
        number_of_results = last_test_part_results_.size();
        get_result = [this](int index) { return last_test_part_results_[index]; };
      }
      else {
        MPI_Recv(&number_of_results, 1, MPI_INT, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        get_result = [rank](int) { return internal::mpi_receive_string(rank); };
      }

      print_partial_results(rank, number_of_results, get_result);
    }
    else if (rank_ == rank) {
      int num_partial_results = last_test_part_results_.size();
      MPI_Send(&num_partial_results, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

      for (const auto& partial_result : last_test_part_results_)
        internal::mpi_send_string(partial_result, 0);
    }
  }

  makeMasterFailIfAnyoneFailed(test_info);

  MASTER_CALLS_DEFAULT_LISTENER(OnTestEnd, test_info);

  MPI_Barrier(MPI_COMM_WORLD);
}

void MPIListener::OnEnvironmentsTearDownStart(const ::testing::UnitTest& unit_test) {
  MASTER_CALLS_DEFAULT_LISTENER(OnEnvironmentsTearDownStart, unit_test);
}

void MPIListener::OnEnvironmentsTearDownEnd(const ::testing::UnitTest& unit_test) {
  MASTER_CALLS_DEFAULT_LISTENER(OnEnvironmentsTearDownEnd, unit_test);
}

void MPIListener::OnTestProgramEnd(const ::testing::UnitTest& unit_test) {
  MPI_Finalize();
}

bool MPIListener::isMasterRank() const {
  return rank_ == 0;
}

void MPIListener::makeMasterFailIfAnyoneFailed(const ::testing::TestInfo& test_info) const {
  bool is_local_passed = test_info.result()->Passed();

  bool all_tests_passed[world_size_];
  MPI_Gather(&is_local_passed, 1, MPI_BYTE, isMasterRank() ? &all_tests_passed : nullptr, 1, MPI_BYTE, 0,
             MPI_COMM_WORLD);

  if (!isMasterRank())
    return;

  auto all_passed =
      std::all_of(all_tests_passed, all_tests_passed + world_size_, [](bool result) { return result; });

  // exploit this to make the master rank fail if any rank failed
  // as a side-effect it calls OnTestPartResult, but since it just collects error messages,
  // and they have been already printed, it won't generate output in any case
  EXPECT_TRUE(all_passed);
}

namespace internal {
void mpi_send_string(const std::string& message, int to_rank) {
  MPI_Send(message.c_str(), message.size() + 1, MPI_CHAR, to_rank, 0, MPI_COMM_WORLD);
}

std::string mpi_receive_string(int from_rank) {
  MPI_Status status;
  MPI_Probe(from_rank, 0, MPI_COMM_WORLD, &status);

  int message_length;
  MPI_Get_count(&status, MPI_CHAR, &message_length);

  char message_buffer[message_length];
  MPI_Recv(message_buffer, message_length, MPI_CHAR, from_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  return message_buffer;
}
}
