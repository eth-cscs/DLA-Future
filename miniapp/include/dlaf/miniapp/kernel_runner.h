//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include "dlaf/common/timer.h"
#include "dlaf/common/vector.h"
#include "dlaf/types.h"

#ifdef DLAF_WITH_CUDA
#include <cuda_runtime.h>
#endif
namespace dlaf::miniapp {

template <Backend backend>
struct KernelRunner;

template <>
struct KernelRunner<Backend::MC> {
  KernelRunner(SizeType count, SizeType nthreads) noexcept : count_(count), nthreads_(nthreads) {
    threads_.reserve(nthreads_);
  }

  // @pre kernel_task should accept only one argument of type SizeType.
  template <class F>
  double run(F&& kernel_task) noexcept {
    auto task = [count = count_, &kernel_task](int id, int tot) {
      for (SizeType i = id * count / tot; i < (id + 1) * count / tot; ++i) {
        kernel_task(i);
      }
    };

    dlaf::common::Timer<> timeit;

    for (SizeType id = 0; id < nthreads_; ++id)
      threads_.emplace_back(std::async(std::launch::async, task, id, nthreads_));

    for (auto& fut : threads_)
      fut.get();
    threads_.clear();

    return timeit.elapsed() / count_;
  }

private:
  SizeType count_;
  SizeType nthreads_;
  common::internal::vector<std::future<void>> threads_;
};

#ifdef DLAF_WITH_CUDA
template <>
struct KernelRunner<Backend::GPU> {
  KernelRunner(SizeType count, SizeType nstreams) noexcept
      : count_(count), streams_(nstreams), handles_(nstreams) {
    for (SizeType i = 0; i < nstreams; ++i) {
      DLAF_CUDA_CALL(cudaStreamCreate(&streams_[i]));
      DLAF_CUBLAS_CALL(cublasCreate(&handles_[i]));
      DLAF_CUBLAS_CALL(cublasSetStream(handles_[i], streams_[i]));
    }
  }

  ~KernelRunner() noexcept {
    for (auto& handle : handles_)
      DLAF_CUBLAS_CALL(cublasDestroy(handle));
    for (auto& stream : streams_)
      DLAF_CUDA_CALL(cudaStreamDestroy(stream));
  }

  // @pre kernel_task should accept exactly two arguments. First argument of type SizeType,
  // the second of type cudaStream_t.
  template <class F>
  double runStream(F&& kernel_task) noexcept {
    return runInternal(std::forward<F>(kernel_task), streams_);
  }

  // @pre kernel_task should accept exactly two arguments. First argument of type SizeType,
  // the second of type cublasHandle_t.
  template <class F>
  double runHandle(F&& kernel_task) noexcept {
    return runInternal(std::forward<F>(kernel_task), handles_);
  }

private:
  template <class F, class Vector>
  double runInternal(F&& kernel_task, const Vector& v) noexcept {
    dlaf::common::Timer<> timeit;

    for (SizeType i = 0; i < count_; ++i) {
      kernel_task(i, v[i % v.size()]);
    }

    for (auto& stream : streams_)
      DLAF_CUDA_CALL(cudaStreamSynchronize(stream));

    return timeit.elapsed() / count_;
  }

  SizeType count_;
  dlaf::common::internal::vector<cudaStream_t> streams_;
  dlaf::common::internal::vector<cublasHandle_t> handles_;
};

#endif
}
