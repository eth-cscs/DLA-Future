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
#include "dlaf/lapack/tile.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/miniapp/dispatch.h"
#include "dlaf/miniapp/kernel_runner.h"
#include "dlaf/miniapp/options.h"
#include "dlaf/miniapp/work_tiles.h"
#include "dlaf/types.h"

using namespace dlaf;
using namespace dlaf::miniapp;

struct Options : MiniappKernelOptions<SupportReal::Yes, SupportComplex::Yes> {
  SizeType m;
  SizeType n;
  SizeType ld;
  blas::Uplo uplo;

  Options(const pika::program_options::variables_map& vm)
      : MiniappKernelOptions(vm), m(vm["m"].as<SizeType>()), n(vm["n"].as<SizeType>()),
        ld(vm["ld"].as<SizeType>()), uplo(dlaf::miniapp::parseUplo(vm["uplo"].as<std::string>())) {
    DLAF_ASSERT(m > 0, m);
    DLAF_ASSERT(n > 0, n);
    DLAF_ASSERT(ld >= m && ld > 0, ld, m);
  }

  Options(Options&&) = default;
  Options(const Options&) = default;
  Options& operator=(Options&&) = default;
  Options& operator=(const Options&) = default;
};

double memOps(blas::Uplo uplo, double m, double n) {
  if (uplo == blas::Uplo::General)
    return m * n;
  // If upper swap sizes and compute as lower
  if (uplo == blas::Uplo::Upper)
    std::swap(m, n);

  double triangular_size = std::min(m, n);
  return triangular_size * (triangular_size + 1) / 2 + (m - triangular_size) * n;
}

struct Test {
  template <Backend backend, class T>
  static void run(const Options& opts) {
    constexpr Device device = DefaultDevice_v<backend>;

    auto el = [](const TileElementIndex&) { return T{0}; };

    const auto uplo = opts.uplo;
    const SizeType m = opts.m;
    const SizeType n = opts.n;
    const SizeType ld = opts.ld;

    WorkTiles<T, device> tiles(opts.count, m, n, ld);
    const T alpha(1);
    const T beta(-1);

    [[maybe_unused]] auto kernel_MC = [uplo, m, n, alpha, beta, &tiles](SizeType i) {
      lapack::laset(uplo, m, n, alpha, beta, tiles(i).ptr(), tiles(i).ld());
    };
#ifdef DLAF_WITH_CUDA
    [[maybe_unused]] auto kernel_GPU = [uplo, m, n, alpha, beta, &tiles](SizeType i,
                                                                         cudaStream_t stream) {
      gpulapack::laset(util::blasToCublas(uplo), m, n, alpha, beta, tiles(i).ptr(), tiles(i).ld(),
                       stream);
    };
#endif
    const double mem_ops = memOps(uplo, m, n);

    KernelRunner<backend> runner(opts.count, opts.nparallel);

    for (SizeType run_index = 0; run_index < opts.nruns; ++run_index) {
      tiles.setElements(el);

      double elapsed_time = -1;
      if constexpr (backend == Backend::MC) {
        elapsed_time = runner.run(kernel_MC);
      }
#ifdef DLAF_WITH_CUDA
      if constexpr (backend == Backend::GPU) {
        elapsed_time = runner.runStream(kernel_GPU);
      }
#endif
      double bandw = mem_ops * sizeof(T) / elapsed_time / 1e9;

      std::cout << "[" << run_index << "]"
                << " " << elapsed_time << "s"
                << " " << bandw << "GB/s"
                << " " << dlaf::internal::FormatShort{opts.type}
                << dlaf::internal::FormatShort{opts.uplo} << " " << m << " " << n << " " << ld << " "
                << opts.nparallel << " " << backend << std::endl;

      if ((opts.do_check == dlaf::miniapp::CheckIterFreq::Last && run_index == (opts.nruns - 1)) ||
          opts.do_check == dlaf::miniapp::CheckIterFreq::All) {
        auto ref = [uplo, el, alpha, beta](const TileElementIndex& index) {
          if (index.row() == index.col())
            return beta;
          if (index.col() > index.row() && uplo == blas::Uplo::Lower)
            return el(index);
          if (index.col() < index.row() && uplo == blas::Uplo::Upper)
            return el(index);
          return alpha;
        };

        auto error = tiles.check(ref);
        if (error > 1)
          std::cout << "CHECK FAILED!!!: ";

        std::cout << "| res - ref | / | res | / eps: " << error << std::endl;
      }
    }
  }
};

int main(int argc, char** argv) {
  // options
  using namespace pika::program_options;
  options_description desc_commandline("Usage: miniapp_laset [options]");
  desc_commandline.add(getMiniappKernelOptionsDescription());

  // clang-format off
  desc_commandline.add_options()
    ("m",  value<SizeType>() ->default_value(256), "Tile row size")
    ("n",  value<SizeType>() ->default_value(256), "Tile col size")
    ("ld",  value<SizeType>() ->default_value(512), "Tile leading dimension")
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
