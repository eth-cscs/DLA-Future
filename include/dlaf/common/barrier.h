//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <pthread.h>

namespace dlaf {

struct barrier_t {
  barrier_t(const unsigned int nthreads) {
    pthread_barrier_init(&barrier_, NULL, nthreads);
  }

  ~barrier_t() {
    pthread_barrier_destroy(&barrier_);
  }

  void arrive_and_wait() {
    pthread_barrier_wait(&barrier_);
  }

  pthread_barrier_t barrier_;
};

}
