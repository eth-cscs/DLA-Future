// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

class MPIListener : public ::testing::TestEventListener {
public:
  MPIListener(int argc, char** argv, ::testing::TestEventListener* other);

protected:
  virtual void OnTestProgramStart(const ::testing::UnitTest& unit_test) override;
  virtual void OnTestIterationStart(const ::testing::UnitTest& unit_test, int iteration) override;
  virtual void OnEnvironmentsSetUpStart(const ::testing::UnitTest& unit_test) override;
  virtual void OnEnvironmentsSetUpEnd(const ::testing::UnitTest& unit_test) override;
  virtual void OnTestCaseStart(const ::testing::TestCase& test_case) override;
  virtual void OnTestStart(const ::testing::TestInfo& test_info) override;
  virtual void OnTestPartResult(const ::testing::TestPartResult& test_part_result) override;
  virtual void OnTestEnd(const ::testing::TestInfo& test_info) override;
  virtual void OnTestCaseEnd(const ::testing::TestCase& test_case) override;
  virtual void OnEnvironmentsTearDownStart(const ::testing::UnitTest& unit_test) override;
  virtual void OnEnvironmentsTearDownEnd(const ::testing::UnitTest& unit_test) override;
  virtual void OnTestIterationEnd(const ::testing::UnitTest& unit_test, int iteration) override;
  virtual void OnTestProgramEnd(const ::testing::UnitTest& unit_test) override;

private:
  bool isMasterRank() const;
  void OnTestEndAllRanks(const ::testing::TestInfo& test_info) const;

  int rank_;
  int world_size_;

  std::vector<std::string> last_test_part_results_;
  std::unique_ptr<::testing::TestEventListener> listener_;
};
