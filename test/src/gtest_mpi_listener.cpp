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
#include <vector>

MPIListener::MPIListener(int argc, char** argv) : argc_(argc), argv_(argv) {}

void MPIListener::OnTestProgramStart(const ::testing::UnitTest& unit_test) {
  MPI_Init(&argc_, &argv_);

  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
}

void MPIListener::OnTestProgramEnd(const ::testing::UnitTest& unit_test) {
  MPI_Finalize();
}

// Called before a test starts.
void MPIListener::OnTestStart(const ::testing::TestInfo& test_info) {
  last_test_result_ = "";

  if (!isMainRank())
    return;
}

// Called after a failed assertion or a SUCCESS().
void MPIListener::OnTestPartResult(const ::testing::TestPartResult& test_part_result) {
  std::ostringstream error_description;

  error_description << "- " << (test_part_result.failed() ? "Failure" : "Success") << " in "
                    << test_part_result.file_name() << ":" << test_part_result.line_number() << std::endl
                    << test_part_result.summary() << std::endl;

  last_test_result_ += error_description.str();
}

// Called after a test ends.
void MPIListener::OnTestEnd(const ::testing::TestInfo& test_info) {
  int result = test_info.result()->Passed();

  std::vector<int> results(world_size_, 0);
  MPI_Gather(&result, 1, MPI_INT, results.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (isMainRank()) {
    bool final_result = std::all_of(results.begin(), results.end(), [](bool r) { return r; });

    printf("*** Test %s.%s RESULT = %s\n", test_info.test_case_name(), test_info.name(),
           final_result ? "OK" : "FAILED");

  }

  for (int rank = 0; rank < world_size_; rank++) {
    if (isMainRank()) {
      std::string rank_error_message;
      if (!results[rank]) {
        if (rank != 0) {
          MPI_Status status;
          MPI_Probe(rank, 0, MPI_COMM_WORLD, &status);

          int number_amount;
          MPI_Get_count(&status, MPI_CHAR, &number_amount);

          char buffer[number_amount];
          MPI_Recv(buffer, number_amount, MPI_CHAR, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

          rank_error_message = buffer;
        }
        else
          rank_error_message = last_test_result_;

        printf("[R%d]\n%s", rank, rank_error_message.c_str());
      }
    }
    else if (rank_ == rank) {
      if (!result) {
        MPI_Send(last_test_result_.c_str(), last_test_result_.size() + 1, MPI_CHAR, 0, 0,
                 MPI_COMM_WORLD);
      }
    }
  }
}

bool MPIListener::isMainRank() const {
  return rank_ == 0;
}
