// Copyright 2006, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

// dlaf-multi-license-check

#include <cstdio>

#include <pika/init.hpp>
#include <pika/program_options.hpp>

#include <dlaf/init.h>

#include "gtest_mpi_listener.h"

#include <gtest/gtest.h>

GTEST_API_ int test_main(pika::program_options::variables_map& vm) {
  std::printf("Running main() from gtest_mpipika_main.cpp\n");
  auto ret = [&] {
    dlaf::ScopedInitializer init(vm);
    return RUN_ALL_TESTS();
  }();
  pika::finalize();
  return ret;
}

GTEST_API_ int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);

  // Initialize MPI
  int threading_required = MPI_THREAD_MULTIPLE;
  int threading_provided;
  MPI_Init_thread(&argc, &argv, threading_required, &threading_provided);

  if (threading_provided != threading_required) {
    std::fprintf(stderr, "Provided MPI threading model does not match the required one.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

  // Gets hold of the event listener list.
  ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();

  // Note:
  // This is a workaround that, by waiting that all pika tasks are finished
  // at the end of each test, ensures that the blocking MPI calls issued during
  // the collection of results from all the MPI ranks do not create potential
  // deadlock conditions.
  struct MPIPIKAListener : public MPIListener {
    using MPIListener::MPIListener;

  protected:
    virtual void OnTestEnd(const ::testing::TestInfo& test_info) override {
      pika::wait();
      MPIListener::OnTestEnd(test_info);
    }
  };

  // Adds MPIPIKAListener to the end. googletest takes the ownership.
  auto default_listener = listeners.Release(listeners.default_result_printer());
  listeners.Append(new MPIPIKAListener(argc, argv, default_listener));

  using namespace pika::program_options;
  options_description desc_commandline("Usage: <test executable> [options]");
  desc_commandline.add(dlaf::getOptionsDescription());

  pika::init_params p;
  p.desc_cmdline = desc_commandline;

  // Initialize pika
  auto ret = pika::init(test_main, argc, argv, p);

  MPI_Finalize();

  return ret;
}
