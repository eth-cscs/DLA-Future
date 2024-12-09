//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <cstdlib>
#include <iostream>
#include <string>

#include <pika/init.hpp>
#include <pika/program_options.hpp>
#include <pika/runtime.hpp>

#include <dlaf/common/format_short.h>
#include <dlaf/common/index2d.h>
#include <dlaf/common/range2d.h>
#include <dlaf/common/timer.h>
#include <dlaf/communication/communicator_grid.h>
#include <dlaf/communication/init.h>
#include <dlaf/eigensolver/bt_reduction_to_band.h>
#include <dlaf/init.h>
#include <dlaf/matrix/copy.h>
#include <dlaf/matrix/distribution.h>
#include <dlaf/matrix/index.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/matrix_mirror.h>
#include <dlaf/matrix/matrix_ref.h>
#include <dlaf/miniapp/dispatch.h>
#include <dlaf/miniapp/options.h>
#include <dlaf/types.h>
#include <dlaf/util_math.h>
#include <dlaf/util_matrix.h>

namespace {
using dlaf::Backend;
using dlaf::DefaultDevice_v;
using dlaf::Device;
using dlaf::GlobalElementIndex;
using dlaf::GlobalElementSize;
using dlaf::Matrix;
using dlaf::SizeType;
using dlaf::TileElementSize;
using dlaf::comm::Communicator;
using dlaf::comm::CommunicatorGrid;
using dlaf::comm::Index2D;
using dlaf::comm::Size2D;
using dlaf::common::Ordering;
using dlaf::matrix::Distribution;
using dlaf::matrix::MatrixMirror;
using dlaf::matrix::util::set;

struct Options
    : dlaf::miniapp::MiniappOptions<dlaf::miniapp::SupportReal::Yes, dlaf::miniapp::SupportComplex::Yes> {
  SizeType m;
  SizeType n;
  SizeType mb;
  SizeType nb;
  SizeType b;
  SizeType eval_idx_end;

  Options(const pika::program_options::variables_map& vm)
      : MiniappOptions(vm), m(vm["m"].as<SizeType>()), n(vm["n"].as<SizeType>()),
        mb(vm["mb"].as<SizeType>()), nb(vm["nb"].as<SizeType>()), b(vm["b"].as<SizeType>()) {
    DLAF_ASSERT(m > 0, m);
    DLAF_ASSERT(mb > 0, mb);

    if (vm.count("percent-evals") == 1 && vm.count("eval-index-end") == 1) {
      std::cerr << "ERROR! "
                   "You can't specify both --percent-evals and --eval-index-end at the same time."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }

    if (vm.count("percent-evals") == 1) {
      double percent = vm["percent-evals"].as<double>();
      eval_idx_end = dlaf::util::internal::percent_to_index(n, percent);
    }
    else if (vm.count("eval-index-end") == 1) {
      eval_idx_end = vm["eval-index-end"].as<SizeType>();
    }
    else {
      eval_idx_end = n;
    }

    DLAF_ASSERT(eval_idx_end >= 0 && eval_idx_end <= n, eval_idx_end);

    if (do_check != dlaf::miniapp::CheckIterFreq::None) {
      std::cerr << "Warning! At the moment result checking it is not implemented." << std::endl;
      do_check = dlaf::miniapp::CheckIterFreq::None;
    }
  }

  Options(Options&&) = default;
  Options(const Options&) = default;
  Options& operator=(Options&&) = default;
  Options& operator=(const Options&) = default;
};
}

struct BacktransformBandToTridiagMiniapp {
  template <Backend backend, typename T>
  static void run(const Options& opts) {
    using MatrixMirrorType = MatrixMirror<T, DefaultDevice_v<backend>, Device::CPU>;
    using HostMatrixType = Matrix<T, Device::CPU>;
    using ConstHostMatrixType = Matrix<const T, Device::CPU>;
    using MatrixRefType = dlaf::matrix::internal::MatrixRef<T, DefaultDevice_v<backend>>;

    Communicator world(MPI_COMM_WORLD);
    CommunicatorGrid comm_grid(world, opts.grid_rows, opts.grid_cols, Ordering::ColumnMajor);

    // Construct reference matrices.
    GlobalElementSize mat_e_size(opts.m, opts.n);
    TileElementSize mat_e_block_size(opts.mb, opts.nb);

    ConstHostMatrixType mat_e_ref = [mat_e_size, mat_e_block_size, &comm_grid]() {
      using dlaf::matrix::util::set_random;

      HostMatrixType random(mat_e_size, mat_e_block_size, comm_grid);
      set_random(random);

      return random;
    }();

    GlobalElementSize mat_hh_size(opts.m, opts.m);
    TileElementSize mat_hh_block_size(opts.mb, opts.mb);

    // Note: random HHRs are not correct, but do not influence the benchmark result.
    ConstHostMatrixType mat_hh_ref = [mat_hh_size, mat_hh_block_size, &comm_grid]() {
      using dlaf::matrix::util::set_random;

      HostMatrixType random(mat_hh_size, mat_hh_block_size, comm_grid);
      set_random(random);

      return random;
    }();

    auto nr_reflectors = std::max<SizeType>(0, opts.m - opts.b - 1);

    Matrix<T, Device::CPU> mat_taus(Distribution(GlobalElementSize(nr_reflectors, 1),
                                                 TileElementSize(opts.mb, 1),
                                                 Size2D(comm_grid.size().cols(), 1),
                                                 Index2D(comm_grid.rank().col(), 0), Index2D(0, 0)));
    set(mat_taus, [](const GlobalElementIndex&) { return T(2); });

    for (int64_t run_index = -opts.nwarmups; run_index < opts.nruns; ++run_index) {
      if (0 == world.rank() && run_index >= 0)
        std::cout << "[" << run_index << "]" << std::endl;

      HostMatrixType mat_e_host(mat_e_size, mat_e_block_size, comm_grid);
      copy(mat_e_ref, mat_e_host);

      HostMatrixType mat_hh_host(mat_hh_size, mat_hh_block_size, comm_grid);
      copy(mat_hh_ref, mat_hh_host);

      double elapsed_time;
      {
        MatrixMirrorType mat_e(mat_e_host);
        MatrixMirrorType mat_hh(mat_hh_host);

        // Wait for matrices to be copied
        mat_e.get().waitLocalTiles();
        mat_hh.get().waitLocalTiles();
        mat_taus.waitLocalTiles();
        DLAF_MPI_CHECK_ERROR(MPI_Barrier(world));

        auto spec =
            dlaf::matrix::util::internal::sub_matrix_spec_slice_cols(mat_e_host, 0, opts.eval_idx_end);
        MatrixRefType mat_e_ref(mat_e.get(), spec);

        dlaf::common::Timer<> timeit;
        if (opts.local)
          dlaf::eigensolver::internal::bt_reduction_to_band<backend, DefaultDevice_v<backend>, T>(
              opts.b, mat_e_ref, mat_hh.get(), mat_taus);
        else
          dlaf::eigensolver::internal::bt_reduction_to_band<backend, DefaultDevice_v<backend>, T>(
              comm_grid, opts.b, mat_e_ref, mat_hh.get(), mat_taus);

        // wait and barrier for all ranks
        mat_e.get().waitLocalTiles();
        comm_grid.wait_all_communicators();

        elapsed_time = timeit.elapsed();
      }

      double gigaflops;
      {
        const double m = mat_e_host.size().rows();
        const double n = opts.eval_idx_end;
        auto add_mul = (m - opts.b) * (m - opts.b) * n;
        gigaflops = dlaf::total_ops<T>(add_mul, add_mul) / elapsed_time / 1e9;
      }

      // print benchmark results
      if (0 == world.rank() && run_index >= 0) {
        std::cout << "[" << run_index << "]"
                  << " " << elapsed_time << "s"
                  << " " << gigaflops << "GFlop/s"
                  << " " << dlaf::internal::FormatShort{opts.type} << " " << mat_e_host.size() << " ("
                  << 0l << ", " << opts.eval_idx_end << ") " << mat_e_host.blockSize() << " " << opts.b
                  << " " << comm_grid.size() << " " << pika::get_os_thread_count() << " " << backend
                  << std::endl;
        if (opts.csv_output) {
          // CSV formatted output with column names that can be read by pandas to simplify
          // post-processing CSVData{-version}, value_0, title_0, value_1, title_1
          std::cout << "CSVData-2, "
                    << "run, " << run_index << ", "
                    << "time, " << elapsed_time << ", "
                    << "GFlops, " << gigaflops << ", "
                    << "type, " << dlaf::internal::FormatShort{opts.type}.value << ", "
                    << "matrixsize, " << mat_e_host.size().rows() << ", "
                    << "blocksize, " << mat_e_host.blockSize().rows() << ", "
                    << "band_size, " << opts.b << ", "
                    << "comm_rows, " << comm_grid.size().rows() << ", "
                    << "comm_cols, " << comm_grid.size().cols() << ", "
                    << "threads, " << pika::get_os_thread_count() << ", "
                    << "backend, " << backend << ", "
                    << "eigenvalue index begin, " << 0l << ", "
                    << "eigenvalue index end, " << opts.eval_idx_end << ", " << opts.info << std::endl;
        }
      }
      // (optional) run test
      if ((opts.do_check == dlaf::miniapp::CheckIterFreq::Last && run_index == (opts.nruns - 1)) ||
          opts.do_check == dlaf::miniapp::CheckIterFreq::All) {
        DLAF_UNIMPLEMENTED("Check");
      }
    }
  }
};

int pika_main(pika::program_options::variables_map& vm) {
  pika::scoped_finalize pika_finalizer;
  dlaf::ScopedInitializer init(vm);
  const Options opts(vm);

  dlaf::miniapp::dispatchMiniapp<BacktransformBandToTridiagMiniapp>(opts);

  return EXIT_SUCCESS;
}

int main(int argc, char** argv) {
  // Init MPI
  dlaf::comm::mpi_init mpi_initter(argc, argv);

  // options
  using namespace pika::program_options;
  options_description desc_commandline("Usage: miniapp_bt_band_to_tridiag [options]");
  desc_commandline.add(dlaf::miniapp::getMiniappOptionsDescription());
  desc_commandline.add(dlaf::getOptionsDescription());

  // clang-format off
  desc_commandline.add_options()
    ("m",              value<SizeType>() ->default_value(2048), "Matrix E rows")
    ("n",              value<SizeType>() ->default_value(4096), "Matrix E columns")
    ("mb",             value<SizeType>() ->default_value( 256), "Matrix E block rows")
    ("nb",             value<SizeType>() ->default_value( 512), "Matrix E block columns")
    ("b",              value<SizeType>() ->default_value(  64), "Band size")
    ("eval-index-end", value<SizeType>()                      , "Index of last eigenvalue of interest/eigenvector to transform")
    ("percent-evals",  value<double>()                        , "Percentage of eigenvalues of interest/eigenvectors to transform")
  ;
  // clang-format on
  dlaf::miniapp::addUploOption(desc_commandline);

  pika::init_params p;
  p.desc_cmdline = desc_commandline;
  return pika::init(pika_main, argc, argv, p);
}
