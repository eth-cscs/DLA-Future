//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <algorithm>
#include <cstddef>
#include <functional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "gtest_mpi_listener.h"

#define MASTER_CALLS_DEFAULT_LISTENER(name, ...) \
  if (isMasterRank())                            \
    listener_->name(__VA_ARGS__);

namespace internal {
void mpi_send_string(const std::string& message, int to_rank);
std::string mpi_receive_string(int from_rank);
}

MPIListener::MPIListener(int, char**, ::testing::TestEventListener* other)
    : listener_(std::move(other)) {
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
  last_test_part_results_.clear();
}

void MPIListener::OnTestPartResult(const ::testing::TestPartResult& test_part_result) {
  if (test_part_result.fatally_failed()) {
    MASTER_CALLS_DEFAULT_LISTENER(OnTestPartResult, test_part_result);
    return;
  }

  std::ostringstream os;
  os << test_part_result;
  last_test_part_results_.push_back(os.str());
}

void MPIListener::OnTestEnd(const ::testing::TestInfo& test_info) {
  auto print_partial_results = [](int rank, int total_results,
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
        number_of_results = static_cast<int>(last_test_part_results_.size());
        get_result = [this](int index) {
          return last_test_part_results_[static_cast<std::size_t>(index)];
        };
      }
      else {
        MPI_Recv(&number_of_results, 1, MPI_INT, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        get_result = [rank](int) { return internal::mpi_receive_string(rank); };
      }

      print_partial_results(rank, number_of_results, get_result);
    }
    else if (rank_ == rank) {
      int num_partial_results = static_cast<int>(last_test_part_results_.size());
      MPI_Send(&num_partial_results, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

      for (const auto& partial_result : last_test_part_results_)
        internal::mpi_send_string(partial_result, 0);
    }
  }

  OnTestEndAllRanks(test_info);

  MPI_Barrier(MPI_COMM_WORLD);
}

void MPIListener::OnTestCaseEnd(const ::testing::TestCase& test_case) {
  MASTER_CALLS_DEFAULT_LISTENER(OnTestCaseEnd, test_case);
}

void MPIListener::OnEnvironmentsTearDownStart(const ::testing::UnitTest& unit_test) {
  MASTER_CALLS_DEFAULT_LISTENER(OnEnvironmentsTearDownStart, unit_test);
}

void MPIListener::OnEnvironmentsTearDownEnd(const ::testing::UnitTest& unit_test) {
  MASTER_CALLS_DEFAULT_LISTENER(OnEnvironmentsTearDownEnd, unit_test);
}

void MPIListener::OnTestIterationEnd(const ::testing::UnitTest& unit_test, int iteration) {
  MASTER_CALLS_DEFAULT_LISTENER(OnTestIterationEnd, unit_test, iteration);
}

void MPIListener::OnTestProgramEnd(const ::testing::UnitTest& unit_test) {
  MASTER_CALLS_DEFAULT_LISTENER(OnTestProgramEnd, unit_test);
}

bool MPIListener::isMasterRank() const {
  return rank_ == 0;
}

void MPIListener::OnTestEndAllRanks(const ::testing::TestInfo& test_info) const {
  bool is_local_passed = test_info.result()->Passed();

  auto all_tests_passed = std::make_unique<bool[]>(static_cast<std::size_t>(world_size_));
  MPI_Gather(&is_local_passed, 1, MPI_BYTE, isMasterRank() ? all_tests_passed.get() : nullptr, 1,
             MPI_BYTE, 0, MPI_COMM_WORLD);

  if (!isMasterRank())
    return;

  auto how_many_ranks_failed =
      std::count(all_tests_passed.get(), all_tests_passed.get() + world_size_, false);

  // exploit this to make the master rank fail if any rank failed
  // as a side-effect it calls OnTestPartResult, but since it just collects error messages,
  // and they have been already printed, it won't generate output in any case
  EXPECT_EQ(0, how_many_ranks_failed);

  MASTER_CALLS_DEFAULT_LISTENER(OnTestEnd, test_info);

  if (how_many_ranks_failed == 0)
    return;

  printf("[  INFO    ] %ld of %d ranks failed\n", how_many_ranks_failed, world_size_);
  fflush(stdout);
}

namespace internal {
void mpi_send_string(const std::string& message, int to_rank) {
// -Wtype-safety disabled because of false positive warning in clang:
// https://github.com/llvm/llvm-project/issues/62512
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wtype-safety"
#endif
  MPI_Send(message.c_str(), static_cast<int>(message.size()) + 1, MPI_CHAR, to_rank, 0, MPI_COMM_WORLD);
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
}

std::string mpi_receive_string(int from_rank) {
  MPI_Status status;
  MPI_Probe(from_rank, 0, MPI_COMM_WORLD, &status);

  int message_length;
  MPI_Get_count(&status, MPI_CHAR, &message_length);

  std::vector<char> message_buffer(static_cast<std::size_t>(message_length));
  MPI_Recv(message_buffer.data(), message_length, MPI_CHAR, from_rank, 0, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);

  return message_buffer.data();
}
}
