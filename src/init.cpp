//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2020, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <dlaf/common/assert.h>
#include <dlaf/init.h>

#ifdef DLAF_WITH_CUDA
#include <dlaf/cublas/executor.h>
#include <dlaf/cuda/executor.h>
#endif

#include <memory>

namespace dlaf {
namespace internal {
bool& initialized() {
  static bool i = false;
  return i;
}

#ifdef DLAF_WITH_CUDA
static std::unique_ptr<cuda::StreamPool> np_stream_pool{nullptr};

void initializeNpCudaStreamPool() {
  DLAF_ASSERT(!np_stream_pool, "");
  np_stream_pool = std::make_unique<cuda::StreamPool>(0, 3, hpx::threads::thread_priority::normal);
}

void finalizeNpCudaStreamPool() {
  DLAF_ASSERT(bool(np_stream_pool), "");
  np_stream_pool.reset();
}

cuda::StreamPool getNpCudaStreamPool() {
  DLAF_ASSERT(bool(np_stream_pool), "");
  return *np_stream_pool;
}

static std::unique_ptr<cuda::StreamPool> hp_stream_pool{nullptr};

void initializeHpCudaStreamPool() {
  DLAF_ASSERT(!hp_stream_pool, "");
  hp_stream_pool = std::make_unique<cuda::StreamPool>(0, 3, hpx::threads::thread_priority::high);
}

void finalizeHpCudaStreamPool() {
  DLAF_ASSERT(bool(hp_stream_pool), "");
  hp_stream_pool.reset();
}

cuda::StreamPool getHpCudaStreamPool() {
  DLAF_ASSERT(bool(hp_stream_pool), "");
  return *hp_stream_pool;
}

static std::unique_ptr<cublas::HandlePool> handle_pool{nullptr};

void initializeCublasHandlePool() {
  DLAF_ASSERT(!handle_pool, "");
  handle_pool = std::make_unique<cublas::HandlePool>(0, CUBLAS_POINTER_MODE_HOST);
}

void finalizeCublasHandlePool() {
  DLAF_ASSERT(bool(handle_pool), "");
  handle_pool.reset();
}

cublas::HandlePool getCublasHandlePool() {
  DLAF_ASSERT(bool(handle_pool), "");
  return *handle_pool;
}
#endif
}
}
