//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <future>

#include "dlaf/blas/enum_output.h"
#include "dlaf/common/format_short.h"
#include "dlaf/lapack/gpu/larft.h"
#include "dlaf/lapack/tile.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/miniapp/dispatch.h"
#include "dlaf/miniapp/kernel_runner.h"
#include "dlaf/miniapp/options.h"
#include "dlaf/miniapp/work_tiles.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

using namespace dlaf;
using namespace dlaf::miniapp;
using dlaf::matrix::test::createTile;
using dlaf::matrix::util::internal::getter_random;

struct Options : MiniappKernelOptions<SupportReal::Yes, SupportComplex::Yes> {
  SizeType n;
  SizeType k;
  SizeType ldv;
  SizeType ldt;

  Options(const pika::program_options::variables_map& vm)
      : MiniappKernelOptions(vm), n(vm["n"].as<SizeType>()), k(vm["k"].as<SizeType>()),
        ldv(vm["ldv"].as<SizeType>()), ldt(vm["ldt"].as<SizeType>()) {
    DLAF_ASSERT(n > 0, n);
    DLAF_ASSERT(k > 0, k);
    // Limit the benchmarks to relevant cases.
    DLAF_ASSERT(k <= n, k, n);
    DLAF_ASSERT(ldv >= n && ldv > 0, ldv, n);
    DLAF_ASSERT(ldt >= k && ldt > 0, ldt, k);
  }

  Options(Options&&) = default;
  Options(const Options&) = default;
  Options& operator=(Options&&) = default;
  Options& operator=(const Options&) = default;
};

template <class T>
double ops(const double n, const double k) {
  DLAF_ASSERT(k <= n, k, n);
  double add = n * k * k / 2 - k * k * k / 6 - n * k / 2;
  double mul = add + k * k / 2;
  return dlaf::total_ops<T>(add, mul);
}

struct Test {
  template <Backend backend, class T>
  static void run(const Options& opts) {
    constexpr Device device = DefaultDevice_v<backend>;

    const SizeType n = opts.n;
    const SizeType k = opts.k;
    const SizeType ldv = opts.ldv;
    const SizeType ldt = opts.ldt;

    // Note: GPU implementation requires the first reflector element to be set to 1.
    getter_random<T> random_value(25698);
    auto el_v = [&random_value](const TileElementIndex& index) {
      if (index.row() == index.col())
        return T{1};
      return random_value();
    };
    auto v = createTile<T, Device::CPU>(el_v, {n, k}, ldv);
    auto tau = createTile<T, Device::CPU>({k, 1}, k);
    for (SizeType j = 0; j < k && j < n; ++j) {
      const auto norm = blas::nrm2(n - j, v.ptr({j, j}), 1);
      tau({j, 0}) = 2 / (norm * norm);
    }

    // As the lower part of t is untouched set it to 0 to allow comparison with GPU implementation.
    auto el_t_cpu = [](const TileElementIndex& index) {
      if (index.row() > index.col())
        return T{0};
      return T{.5};
    };
    // As t should be overwritten completely (lower part is 0 while upper is the result)
    // give some initial value different than 0.
    [[maybe_unused]] auto el_t_gpu = [](const TileElementIndex&) { return T{.5}; };

    WorkTiles<T, device> vs(opts.count, n, k, ldv);
    WorkTiles<T, Device::CPU> taus(opts.count, k, 1, k);
    WorkTiles<T, device> ts(opts.count, k, k, ldt);

    vs.setElementsFromTile(v);
    taus.setElementsFromTile(tau);

    [[maybe_unused]] auto kernel_MC = [n, k, &vs, &taus, &ts](SizeType i) {
      lapack::larft(lapack::Direction::Forward, lapack::StoreV::Columnwise, n, k, vs(i).ptr(),
                    vs(i).ld(), taus(i).ptr(), ts(i).ptr(), ts(i).ld());
    };
#ifdef DLAF_WITH_CUDA
    [[maybe_unused]] auto kernel_GPU0 = [n, k, &vs, &taus, &ts](SizeType i, cublasHandle_t handle) {
      gpulapack::larft0(handle, n, k, vs(i).ptr(), vs(i).ld(), taus(i).ptr(), ts(i).ptr(), ts(i).ld());
    };

    [[maybe_unused]] auto kernel_GPU = [n, k, &vs, &taus, &ts](SizeType i, cublasHandle_t handle) {
      gpulapack::larft(handle, n, k, vs(i).ptr(), vs(i).ld(), taus(i).ptr(), ts(i).ptr(), ts(i).ld());
    };
#endif
    const double flop = ops<T>(n, k);

    KernelRunner<backend> runner(opts.count, opts.nparallel);

    for (SizeType run_index = 0; run_index < opts.nruns; ++run_index) {
      double elapsed_time = -1;
      if constexpr (backend == Backend::MC) {
        ts.setElements(el_t_cpu);
        elapsed_time = runner.run(kernel_MC);
      }
#ifdef DLAF_WITH_CUDA
      if constexpr (backend == Backend::GPU) {
        ts.setElements(el_t_gpu);
        double elapsed_time0 = runner.runHandle(kernel_GPU0);

        std::cout << "[" << run_index << "]"
                  << " " << elapsed_time0 << "s" << std::endl;

        ts.setElements(el_t_gpu);
        elapsed_time = runner.runHandle(kernel_GPU);
      }
#endif
      const double gflops = flop / elapsed_time / 1e9;

      std::cout << "[" << run_index << "]"
                << " " << elapsed_time << "s"
                << " " << gflops << "GFlop/s"
                << " " << dlaf::internal::FormatShort{opts.type} << " " << n << " " << k << " " << ldv
                << " " << ldt << " " << opts.nparallel << " " << backend << std::endl;

      if ((opts.do_check == dlaf::miniapp::CheckIterFreq::Last && run_index == (opts.nruns - 1)) ||
          opts.do_check == dlaf::miniapp::CheckIterFreq::All) {
        auto t = createTile<T, Device::CPU>(el_t_cpu, {k, k}, k);
        lapack::larft(lapack::Direction::Forward, lapack::StoreV::Columnwise, n, k, v.ptr(), v.ld(),
                      tau.ptr(), t.ptr(), t.ld());

        auto error = ts.check(t);
        if (error > k)
          std::cout << "CHECK FAILED!!!: ";

        std::cout << "| res - ref | / | res | / eps: " << error << std::endl;
      }
    }
  }
};

int main(int argc, char** argv) {
  // options
  using namespace pika::program_options;
  options_description desc_commandline("Usage: miniapp_larft [options]");
  desc_commandline.add(getMiniappKernelOptionsDescription());

  // clang-format off
  desc_commandline.add_options()
    ("n",  value<SizeType>() ->default_value(256), "Tile row size")
    ("k",  value<SizeType>() ->default_value(256), "Tile col size")
    ("ldv",  value<SizeType>() ->default_value(512), "Tile leading dimension")
    ("ldt",  value<SizeType>() ->default_value(512), "Tile leading dimension")
  ;
  // clang-format on
  addUploOption(desc_commandline);

  variables_map vm;
  store(parse_command_line(argc, argv, desc_commandline), vm);
  notify(vm);
  if (vm.count("help")) {
    std::cout << desc_commandline << "\n";
    return 1;
  }
  Options options(vm);

  dispatchMiniapp<Test>(options);

  return 0;
}
