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

#include <dlaf/blas/enum_output.h>
#include <dlaf/common/format_short.h>
#include <dlaf/lapack/gpu/larft.h>
#include <dlaf/lapack/tile.h>
#include <dlaf/matrix/tile.h>
#include <dlaf/miniapp/dispatch.h>
#include <dlaf/miniapp/kernel_runner.h>
#include <dlaf/miniapp/options.h>
#include <dlaf/miniapp/work_tiles.h>
#include <dlaf/types.h>
#include <dlaf/util_matrix.h>

#include <dlaf_test/matrix/util_tile.h>

// TODO remove
#include <dlaf/blas/tile_extensions.h>
#include <dlaf/matrix/print_csv.h>
#include <dlaf/matrix/print_gpu.h>

using namespace dlaf;
using namespace dlaf::miniapp;
using dlaf::matrix::test::createTile;
using dlaf::matrix::util::internal::getter_random;

struct Options : MiniappKernelOptions<SupportReal::Yes, SupportComplex::Yes> {
  SizeType m;
  SizeType n;
  SizeType k;
  SizeType ldv;
  SizeType ldt;
  SizeType kernel_id;

  Options(const pika::program_options::variables_map& vm)
      : MiniappKernelOptions(vm), m(vm["m"].as<SizeType>()), n(vm["n"].as<SizeType>()),
        k(vm["k"].as<SizeType>()), ldv(vm["ldv"].as<SizeType>()), ldt(vm["ldt"].as<SizeType>()),
        kernel_id(vm["kernel_id"].as<SizeType>()) {
    DLAF_ASSERT(m > 0, m);
    DLAF_ASSERT(n > 0, n);
    DLAF_ASSERT(k > 0, k);
    // Limit the benchmarks to relevant cases.
    DLAF_ASSERT(k <= n, k, n);
    DLAF_ASSERT(ldv >= m && ldv > 0, ldv, m);
    DLAF_ASSERT(ldt >= k && ldt > 0, ldt, k);
  }

  Options(Options&&) = default;
  Options(const Options&) = default;
  Options& operator=(Options&&) = default;
  Options& operator=(const Options&) = default;
};

template <class T>
double ops(const double m, const double k) {
  if (m == 0)
    return 0;
  double add_mul = (m + 1) * k * (k - 1) / 2;
  return dlaf::total_ops<T>(add_mul, add_mul);
}

template <class T>
void MC_reference(const SizeType m, const SizeType k, const T* v, const SizeType ldv, const T* tau, T* t,
                  const SizeType ldt) {
  for (int j = 1; j < k; ++j) {
    auto v_ = [v, ldv](SizeType i, SizeType j) { return v + i + j * ldv; };
    auto t_ = [t, ldt](SizeType i, SizeType j) { return t + i + j * ldt; };
    blas::gemv(blas::Layout::ColMajor, blas::Op::ConjTrans, m, j, -tau[j], v_(0, 0), ldv, v_(0, j), 1,
               T{1}, t_(0, j), 1);
  }
  for (int j = 0; j < k; ++j) {
    t[j * (ldt + 1)] = tau[j];
  }
}

struct Test {
  template <Backend backend, class T>
  static void run(const Options& opts) {
    constexpr Device device = DefaultDevice_v<backend>;

    const SizeType m = opts.m;
    const SizeType n = opts.n;
    const SizeType k = opts.k;
    const SizeType ldv = opts.ldv;
    const SizeType ldt = opts.ldt;

    getter_random<T> random_value(25698);
    auto rnd = [&random_value](const TileElementIndex&) { return random_value(); };
    auto tau = createTile<T, Device::CPU>(rnd, {k, 1}, k);
    auto v = createTile<T, Device::CPU>(rnd, {m, k}, ldv);

    // As the kernels need T to be zeroed we start from there.
    auto el_t = [](const TileElementIndex&) {
      return T{0.};
    };

    WorkTiles<T, device> vs(opts.count, m, k, ldv);
    WorkTiles<T, device> ts(opts.count, k, k, ldt);
    WorkTiles<T, device> taus(opts.count, k, 1, k);

    vs.setElementsFromTile(v);
    ts.setElements(el_t);

    [[maybe_unused]] auto kernel_MC = [m, k, &vs, &tau, &ts](SizeType i) {
      MC_reference(m, k, vs(i).ptr(), vs(i).ld(), tau.ptr(), ts(i).ptr(), ts(i).ld());
    };
#ifdef DLAF_WITH_CUDA
    [[maybe_unused]] auto kernel_GPU0 = [m, k, &vs, &tau, &ts](SizeType i, cublasHandle_t handle) {
      gpulapack::larft_gemv0(handle, m, k, vs(i).ptr(), vs(i).ld(), tau.ptr(), ts(i).ptr(), ts(i).ld());
    };

    [[maybe_unused]] auto copy_tau = [k, &tau, &taus](SizeType i, whip::stream_t stream) {
      gpulapack::lacpy(blas::Uplo::General, k, 1, tau.ptr(), tau.ld(), taus(i).ptr(), taus(i).ld(), stream);
    };
    [[maybe_unused]] auto kernel_GPU1 = [m, k, &vs, &ts](SizeType i, cublasHandle_t handle) {
      gpulapack::larft_gemv1_notau(handle, m, k, vs(i).ptr(), vs(i).ld(), ts(i).ptr(), ts(i).ld());
    };
    [[maybe_unused]] auto post_kernel_GPU1 = [m, k, &taus, &ts](SizeType i, whip::stream_t stream) {
      gpulapack::larft_gemv1_fixtau(k, taus(i).ptr(), 1, ts(i).ptr(), ts(i).ld(), stream);
    };

    [[maybe_unused]] auto copy_tau_in_t = [k, &tau, &ts](SizeType i, whip::stream_t stream) {
      gpulapack::lacpy(blas::Uplo::General, 1, k, tau.ptr(), 1, ts(i).ptr(), ts(i).ld() + 1, stream);
    };
#define KERNEL_GPU(id)                                                                         \
  [[maybe_unused]] auto kernel_GPU##id = [m, k, &vs, &ts](SizeType i, whip::stream_t stream) {   \
    gpulapack::larft_gemv##id(m, k, vs(i).ptr(), vs(i).ld(), ts(i).ptr(), ts(i).ld(), stream); \
  }

#define EXPAND(macro) \
  macro(1000); \
  macro(1001); \
  macro(1002); \
  macro(1003); \
  macro(1100); \
  macro(1101); \
  macro(1200); \
  macro(1201); \
  macro(1202); \
  macro(1203); \
  macro(1204); \
  macro(2000); \
  macro(2001); \
  macro(2002); \
  macro(2003); \
  macro(2100); \
  macro(2101); \
  macro(2102); \
  macro(2103); \
  macro(2104); \
  macro(2200); \
  macro(2201); \
  macro(2202); \
  macro(2203); \
  macro(2204); \
  macro(2205); \
  macro(2206); \
  macro(2207); \
  macro(2208); \
  macro(2209); \
  macro(2210)

    EXPAND(KERNEL_GPU);

#endif
    const double flop = ops<T>(n, k);

    KernelRunner<backend> runner(opts.count, opts.nparallel);

    for (SizeType run_index = 0; run_index < opts.nruns; ++run_index) {
      double elapsed_time_pre = 0;
      double elapsed_time_kernel = 0;
      double elapsed_time_post = 0;
      ts.setElements(el_t);

      if constexpr (backend == Backend::MC) {
        elapsed_time_kernel = runner.run(kernel_MC);
      }
#ifdef DLAF_WITH_CUDA

#define KERNEL_CASE(id)                              \
  case id:                                           \
    elapsed_time_pre = runner.runStream(copy_tau_in_t); \
    elapsed_time_kernel = runner.runStream(kernel_GPU##id); \
    break;

      if constexpr (backend == Backend::GPU) {
        switch (opts.kernel_id) {
          EXPAND(KERNEL_CASE);
          case 0:
            elapsed_time_kernel = runner.runHandle(kernel_GPU0);
            elapsed_time_post = runner.runStream(copy_tau_in_t); \
            break;
          case 1:
            elapsed_time_pre = runner.runStream(copy_tau);
            elapsed_time_kernel = runner.runHandle(kernel_GPU1);
            elapsed_time_post = runner.runStream(post_kernel_GPU1);
            break;
          default:
            std::cout << "Error: Non existing kernel id" << opts.kernel_id << std::endl;
            elapsed_time_pre = -1;
            elapsed_time_kernel = -1;
            elapsed_time_post = -1;
        }
      }
#endif
      double elapsed_time = elapsed_time_pre + elapsed_time_kernel + elapsed_time_post;
      const double gflops = flop / elapsed_time / 1e9;
      const double gflops_kernel = flop / elapsed_time_kernel / 1e9;

      std::cout << "[" << run_index << "]"
                << " " << elapsed_time << "s"
                << " " << gflops << "GFlop/s"
                << " " << dlaf::internal::FormatShort{opts.type} << " ";
      std::cout << m << " " << k << " " << ldv << " " << ldt << " " << opts.nparallel << " " << backend;
      if (backend == Backend::GPU)
        std::cout << " " << opts.kernel_id;

      std::cout << " |"
                << " PRE: " << elapsed_time_pre << "s"
                << " KERNEL: " << elapsed_time_kernel << "s " << gflops_kernel << "GFlop/s"
                << " POST: " << elapsed_time_post << "s" << std::endl;

      if ((opts.do_check == dlaf::miniapp::CheckIterFreq::Last && run_index == (opts.nruns - 1)) ||
          opts.do_check == dlaf::miniapp::CheckIterFreq::All) {
        auto t = createTile<T, Device::CPU>(el_t, {k, k}, k);
        MC_reference(m, k, v.ptr(), v.ld(), tau.ptr(), t.ptr(), t.ld());

       /*
        print(format::csv{}, ts(0));
        auto tt = createTile<T, Device::CPU>(el_t, {k, k}, k);
        if constexpr (backend == Backend::GPU) {
          whip::stream_t stream = NULL;
          matrix::internal::copy_o(ts(0), tt, stream);
        }
        else
          matrix::internal::copy_o(ts(0), tt);

        tile::internal::add_o(T{-1}, t, tt);
        print(format::csv{}, tt);
        */
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
  options_description desc_commandline("Usage: miniapp_larft_gemv [options]");
  desc_commandline.add(getMiniappKernelOptionsDescription());

  // clang-format off
  desc_commandline.add_options()
    ("m",  value<SizeType>() ->default_value(512), "Tile row size")
    ("n",  value<SizeType>() ->default_value(256), "Tile row size")
    ("k",  value<SizeType>() ->default_value(256), "Tile col size")
    ("ldv",  value<SizeType>() ->default_value(512), "Tile leading dimension")
    ("ldt",  value<SizeType>() ->default_value(512), "Tile leading dimension")
    ("kernel_id",  value<SizeType>() ->default_value(1), "GPU kernel id")
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
