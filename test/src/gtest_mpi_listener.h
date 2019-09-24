// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <memory>

#include <gtest/gtest.h>
#include <mpi.h>

class MPIListener : public ::testing::EmptyTestEventListener {
public:
  MPIListener(int argc, char** argv);

protected:
  virtual void OnTestProgramStart(const ::testing::UnitTest& unit_test) override;
  virtual void OnTestProgramEnd(const ::testing::UnitTest& unit_test) override;

  virtual void OnTestStart(const ::testing::TestInfo& test_info) override;
  virtual void OnTestPartResult(const ::testing::TestPartResult& test_part_result) override;
  virtual void OnTestEnd(const ::testing::TestInfo& test_info) override;

private:
  bool isMainRank() const;

  void printTestFailure(const ::testing::TestPartResult& test_result) const;

  int argc_;
  char** argv_;

  int rank_;
  int world_size_;

  std::string last_test_result_;
};
