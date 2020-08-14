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
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <cstdio>

#include <gtest/gtest.h>
#include <hpx/init.hpp>

#include "gtest_mpi_listener.h"

GTEST_API_ int test_main(int argc, char** argv) {
  std::printf("Running main() from gtest_mpihpx_main.cpp\n");
  auto ret = RUN_ALL_TESTS();
  hpx::finalize();
  return ret;
}

GTEST_API_ int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);

  // Initialize MPI
  int threading_required = MPI_THREAD_SERIALIZED;
  int threading_provided;
  MPI_Init_thread(&argc, &argv, threading_required, &threading_provided);

  if (threading_provided != threading_required) {
    std::fprintf(stderr, "Provided MPI threading model does not match the required one.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // Gets hold of the event listener list.
  ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();

  // Adds MPIListener to the end. googletest takes the ownership.
  auto default_listener = listeners.Release(listeners.default_result_printer());
  listeners.Append(new MPIListener(argc, argv, default_listener));

  // Initialize HPX
  auto ret = hpx::init(test_main, argc, argv);

  MPI_Finalize();

  return ret;
}
