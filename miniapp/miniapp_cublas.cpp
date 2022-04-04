//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <iostream>

#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <pika/future.hpp>
#include <pika/init.hpp>
#include <pika/modules/async_cuda.hpp>
#include <pika/thread.hpp>
#include <pika/unwrap.hpp>

#include "dlaf/executors.h"
#include "dlaf/init.h"
#include "dlaf/sender/policy.h"
#include "dlaf/sender/transform.h"
#include "dlaf/types.h"

int pika_main(int argc, char* argv[]) {
  {
    dlaf::ScopedInitializer init(argc, argv);

    constexpr int n = 10000;
    constexpr int incx = 1;
    constexpr int incy = 1;

    // Initialize buffers on the device
    thrust::device_vector<double> x = thrust::host_vector<double>(n, 4.0);
    thrust::device_vector<double> y = thrust::host_vector<double>(n, 2.0);

    // https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-axpy

    // NOTE: This only serves to produce a future and check that
    // whenAllLift/transform works correctly also with future arguments.
    constexpr double alpha = 2.0;
    pika::future<const double*> alpha_f = pika::make_ready_future(&alpha);

    auto s1 = dlaf::internal::transformLift(dlaf::internal::Policy<dlaf::Backend::GPU>(
                                                pika::threads::thread_priority::high),
                                            cublasDaxpy, n, std::move(alpha_f), x.data().get(), incx,
                                            y.data().get(), incy);

    auto s2 = dlaf::internal::transform(
        dlaf::internal::Policy<dlaf::Backend::MC>(),
        [&y](cublasStatus_t s) {
          DLAF_CUBLAS_CALL(s);

          // Note: This doesn't lead to a race condition because this executes
          // after the `cublasDaxpy()`.
          thrust::host_vector<double> y_h = y;
          double sum_of_elems = 0;
          for (double e : y_h)
            sum_of_elems += e;

          std::cout << "result : " << sum_of_elems << std::endl;
        },
        std::move(s1));

    pika::this_thread::experimental::sync_wait(std::move(s2));
  }

  pika::finalize();

  return 0;
}

int main(int argc, char** argv) {
  return pika::init(pika_main, argc, argv);
}
