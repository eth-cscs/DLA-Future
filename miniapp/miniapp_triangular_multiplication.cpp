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
#include <limits>
#include <string>
#include <type_traits>

#include <blas/util.hh>
#include <mpi.h>

#include <pika/init.hpp>
#include <pika/program_options.hpp>
#include <pika/runtime.hpp>

#include <dlaf/common/format_short.h>
#include <dlaf/common/index2d.h>
#include <dlaf/common/range2d.h>
#include <dlaf/common/timer.h>
#include <dlaf/communication/communicator_grid.h>
#include <dlaf/communication/init.h>
#include <dlaf/init.h>
#include <dlaf/matrix/copy.h>
#include <dlaf/matrix/index.h>
#include <dlaf/matrix/matrix_mirror.h>
#include <dlaf/miniapp/dispatch.h>
#include <dlaf/miniapp/options.h>
#include <dlaf/multiplication/triangular.h>
#include <dlaf/types.h>
#include <dlaf/util_matrix.h>

namespace {

using dlaf::Backend;
using dlaf::DefaultDevice_v;
using dlaf::Device;
using dlaf::GlobalElementIndex;
using dlaf::GlobalElementSize;
using dlaf::SizeType;
using dlaf::TileElementSize;
using dlaf::comm::Communicator;
using dlaf::comm::CommunicatorGrid;
using dlaf::common::Ordering;
using dlaf::matrix::Matrix;
using dlaf::matrix::MatrixMirror;

struct Options
    : dlaf::miniapp::MiniappOptions<dlaf::miniapp::SupportReal::Yes, dlaf::miniapp::SupportComplex::Yes> {
  SizeType m;
  SizeType n;
  SizeType mb;
  SizeType nb;
  blas::Side side;
  blas::Uplo uplo;
  blas::Op op;
  blas::Diag diag;

  Options(const pika::program_options::variables_map& vm)
      : MiniappOptions(vm), m(vm["m"].as<SizeType>()), n(vm["n"].as<SizeType>()),
        mb(vm["mb"].as<SizeType>()), nb(vm["nb"].as<SizeType>()),
        side(dlaf::miniapp::parseSide(vm["side"].as<std::string>())),
        uplo(dlaf::miniapp::parseUplo(vm["uplo"].as<std::string>())),
        op(dlaf::miniapp::parseOp(vm["op"].as<std::string>())),
        diag(dlaf::miniapp::parseDiag(vm["diag"].as<std::string>())) {
    DLAF_ASSERT(m > 0 && n > 0, m, n);
    DLAF_ASSERT(mb > 0 && nb > 0, mb, nb);

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

template <typename T>
using matrix_values_t = std::function<T(const GlobalElementIndex&)>;

template <typename T>
using linear_system_t =
    std::tuple<matrix_values_t<T>, matrix_values_t<T>, matrix_values_t<T>>;  // A, B, X

template <typename T>
linear_system_t<T> sampleLeftTr(blas::Uplo uplo, blas::Op op, blas::Diag diag, T alpha, SizeType m);
}

struct triangularMultiplicationMiniapp {
  template <Backend backend, typename T>
  static void run(const Options& opts) {
    using blas::Side;
    using MatrixMirrorType = MatrixMirror<T, DefaultDevice_v<backend>, Device::CPU>;
    using ConstMatrixMirrorType = MatrixMirror<const T, DefaultDevice_v<backend>, Device::CPU>;
    using HostMatrixType = Matrix<T, Device::CPU>;
    using ConstHostMatrixType = Matrix<const T, Device::CPU>;
    Communicator world(MPI_COMM_WORLD);
    CommunicatorGrid comm_grid(world, opts.grid_rows, opts.grid_cols, Ordering::ColumnMajor);

    const auto side = opts.side;
    const auto uplo = opts.uplo;
    const auto op = opts.op;
    const auto diag = opts.diag;
    const SizeType k = side == Side::Left ? opts.m : opts.n;
    const SizeType kb = side == Side::Left ? opts.mb : opts.nb;

    ConstHostMatrixType ah = [&comm_grid, k, kb]() {
      using dlaf::matrix::util::set_random_non_zero_diagonal;

      GlobalElementSize size(k, k);
      TileElementSize block_size(kb, kb);
      HostMatrixType matrix(size, block_size, comm_grid);
      set_random_non_zero_diagonal(matrix);
      return matrix;
    }();

    GlobalElementSize size_b{opts.m, opts.n};
    TileElementSize block_size_b{opts.mb, opts.nb};

    ConstHostMatrixType b_ref = [&comm_grid, &size_b, &block_size_b]() {
      using dlaf::matrix::util::set_random;

      HostMatrixType matrix(size_b, block_size_b, comm_grid);
      set_random(matrix);
      return matrix;
    }();

    HostMatrixType bh(size_b, block_size_b, comm_grid);

    ConstMatrixMirrorType a(ah);
    MatrixMirrorType b(bh);

    auto sync_barrier = [&]() {
      a.get().waitLocalTiles();
      b.get().waitLocalTiles();
      DLAF_MPI_CHECK_ERROR(MPI_Barrier(world));
    };

    const T alpha = 2.0;

    double m = size_b.rows();
    double n = size_b.cols();
    auto add_mul = n * m * (side == Side::Left ? m : n) / 2;
    const double total_ops = dlaf::total_ops<T>(add_mul, add_mul);

    for (int64_t run_index = -opts.nwarmups; run_index < opts.nruns; ++run_index) {
      if (0 == world.rank() && run_index >= 0)
        std::cout << "[" << run_index << "]" << std::endl;

      using dlaf::matrix::util::set;
      copy(b_ref, bh);
      b.copySourceToTarget();

      sync_barrier();

      dlaf::common::Timer<> timeit;
      if (opts.local)
        dlaf::triangular_multiplication<backend, dlaf::DefaultDevice_v<backend>, T>(side, uplo, op, diag,
                                                                                    alpha, a.get(),
                                                                                    b.get());
      else
        dlaf::triangular_multiplication<backend, dlaf::DefaultDevice_v<backend>, T>(
            comm_grid, side, uplo, op, diag, alpha, a.get(), b.get());

      sync_barrier();

      // benchmark results
      if (0 == world.rank() && run_index >= 0) {
        auto elapsed_time = timeit.elapsed();
        double gigaflops = total_ops / elapsed_time / 1e9;

        std::cout << "[" << run_index << "]" << " " << elapsed_time << "s" << " " << gigaflops 
                << "GFlop/s"
                << " " << dlaf::internal::FormatShort{opts.type}
                << dlaf::internal::FormatShort{opts.side} << dlaf::internal::FormatShort{opts.uplo}
                << dlaf::internal::FormatShort{opts.op} << dlaf::internal::FormatShort{opts.diag}
                << " " << bh.size() << " " << bh.blockSize() << " " << comm_grid.size() << " "
                << pika::get_os_thread_count() << " " << backend << std::endl;
      }

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
  dlaf::miniapp::dispatchMiniapp<triangularMultiplicationMiniapp>(opts);

  return EXIT_SUCCESS;
}

int main(int argc, char** argv) {
  dlaf::comm::mpi_init mpi_initter(argc, argv);

  // options
  using namespace pika::program_options;
  options_description desc_commandline(
      "Benchmark computation of A . B, "
      "where A is a non-unit lower triangular matrix, and B is an m by n matrix\n\n"
      "options\n"
      "Usage: miniapp_triangular_multiplication [options]");
  desc_commandline.add(dlaf::miniapp::getMiniappOptionsDescription());
  desc_commandline.add(dlaf::getOptionsDescription());

  // clang-format off
  desc_commandline.add_options()
    ("m",             value<SizeType>()     ->default_value(4096),       "Matrix b rows")
    ("n",             value<SizeType>()     ->default_value(512),        "Matrix b columns")
    ("mb",            value<SizeType>()     ->default_value(256),        "Matrix b block rows")
    ("nb",            value<SizeType>()     ->default_value(512),        "Matrix b block columns")
  ;
  // clang-format on
  dlaf::miniapp::addSideOption(desc_commandline);
  dlaf::miniapp::addUploOption(desc_commandline);
  dlaf::miniapp::addOpOption(desc_commandline);
  dlaf::miniapp::addDiagOption(desc_commandline);

  pika::init_params p;
  p.desc_cmdline = desc_commandline;
  p.rp_callback = dlaf::initResourcePartitionerHandler;
  return pika::init(pika_main, argc, argv, p);
}
