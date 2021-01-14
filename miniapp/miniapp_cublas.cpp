//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <iostream>

#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <hpx/future.hpp>
#include <hpx/include/util.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/async_cuda.hpp>
#include <hpx/thread.hpp>

#include "dlaf/cublas/executor.h"

int hpx_main() {
  constexpr int device = 0;
  constexpr std::size_t num_streams_per_worker_thread = 10;

  dlaf::cuda::StreamPool stream_pool{device, num_streams_per_worker_thread,
                                     hpx::threads::thread_priority::high};
  dlaf::cublas::HandlePool handle_pool{device, CUBLAS_POINTER_MODE_HOST};
  dlaf::cublas::Executor cublas_exec{stream_pool, handle_pool};

  hpx::cuda::experimental::enable_user_polling p;

  constexpr int n = 10000;
  constexpr int incx = 1;
  constexpr int incy = 1;

  // Initialize buffers on the device
  thrust::device_vector<double> x = thrust::host_vector<double>(n, 4.0);
  thrust::device_vector<double> y = thrust::host_vector<double>(n, 2.0);

  // https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-axpy

  // NOTE: The hpx::async only serves to produce a future and check that
  // dataflow works correctly also with future arguments.
  hpx::future<const double*> alpha_f = hpx::async([]() {
    static constexpr double alpha = 2.0;
    return &alpha;
  });

  hpx::future<cublasStatus_t> f1 = hpx::dataflow(cublas_exec, hpx::util::unwrapping(cublasDaxpy), n,
                                                 alpha_f, x.data().get(), incx, y.data().get(), incy);

  hpx::future<void> f2 = f1.then([&y](hpx::future<cublasStatus_t> s) {
    DLAF_CUBLAS_CALL(s.get());

    // Note: This doesn't lead to a race condition because this executes
    // after the `cublasDaxpy()`.
    thrust::host_vector<double> y_h = y;
    double sum_of_elems = 0;
    for (double e : y_h)
      sum_of_elems += e;

    std::cout << "result : " << sum_of_elems << std::endl;
  });

  f2.get();

  return hpx::finalize();
}

int main(int argc, char** argv) {
  return hpx::init(argc, argv);
}
