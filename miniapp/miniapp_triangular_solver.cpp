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
#include <limits>
#include <type_traits>

#include <mpi.h>
#include <pika/init.hpp>
#include <pika/program_options.hpp>
#include <pika/runtime.hpp>

#include <blas/util.hh>

#include "dlaf/common/format_short.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/range2d.h"
#include "dlaf/common/timer.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/init.h"
#include "dlaf/init.h"
#include "dlaf/matrix/copy.h"
#include "dlaf/matrix/index.h"
#include "dlaf/matrix/matrix_mirror.h"
#include "dlaf/solver.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/util_types.h"

#include "dlaf/miniapp/dispatch.h"
#include "dlaf/miniapp/options.h"

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

struct Options
    : dlaf::miniapp::MiniappOptions<dlaf::miniapp::SupportReal::Yes, dlaf::miniapp::SupportComplex::No> {
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

struct triangularSolverMiniapp {
  template <dlaf::Backend backend, typename T>
  static void run(const Options& opts) {
    Communicator world(MPI_COMM_WORLD);
    CommunicatorGrid comm_grid(world, opts.grid_rows, opts.grid_cols, Ordering::ColumnMajor);

    // Allocate memory for the matrices
    dlaf::matrix::Matrix<T, Device::CPU> ah(GlobalElementSize{opts.m, opts.m},
                                            TileElementSize{opts.mb, opts.mb}, comm_grid);
    dlaf::matrix::Matrix<T, Device::CPU> bh(GlobalElementSize{opts.m, opts.n},
                                            TileElementSize{opts.mb, opts.nb}, comm_grid);

    dlaf::matrix::MatrixMirror<T, DefaultDevice_v<backend>, Device::CPU> a(ah);
    dlaf::matrix::MatrixMirror<T, DefaultDevice_v<backend>, Device::CPU> b(bh);

    auto sync_barrier = [&]() {
      a.get().waitLocalTiles();
      b.get().waitLocalTiles();
      DLAF_MPI_CHECK_ERROR(MPI_Barrier(world));
    };

    const auto side = opts.side;
    const auto uplo = opts.uplo;
    const auto op = opts.op;
    const auto diag = opts.diag;
    const T alpha = 2.0;

    double m = ah.size().rows();
    double n = bh.size().cols();
    auto add_mul = n * m * m / 2;
    const double total_ops = dlaf::total_ops<T>(add_mul, add_mul);

    auto [ref_op_a, ref_b, ref_x] = ::sampleLeftTr(uplo, op, diag, alpha, ah.size().rows());

    for (int64_t run_index = -opts.nwarmups; run_index < opts.nruns; ++run_index) {
      if (0 == world.rank() && run_index >= 0)
        std::cout << "[" << run_index << "]" << std::endl;

      // setup matrix A and b
      using dlaf::matrix::util::set;
      set(ah, ref_op_a, op);
      set(bh, ref_b);
      a.copySourceToTarget();
      b.copySourceToTarget();

      sync_barrier();

      dlaf::common::Timer<> timeit;
      dlaf::solver::triangular<backend, dlaf::DefaultDevice_v<backend>, T>(comm_grid, side, uplo, op,
                                                                           diag, alpha, a.get(),
                                                                           b.get());

      sync_barrier();

      // benchmark results
      if (0 == world.rank() && run_index >= 0) {
        auto elapsed_time = timeit.elapsed();
        double gigaflops = total_ops / elapsed_time / 1e9;

        std::cout << "[" << run_index << "]"
                  << " " << elapsed_time << "s"
                  << " " << gigaflops << "GFlop/s"
                  << " " << dlaf::internal::FormatShort{opts.type}
                  << dlaf::internal::FormatShort{opts.side} << dlaf::internal::FormatShort{opts.uplo}
                  << dlaf::internal::FormatShort{opts.op} << dlaf::internal::FormatShort{opts.diag}
                  << " " << bh.size() << " " << bh.blockSize() << " " << comm_grid.size() << " "
                  << pika::get_os_thread_count() << " " << backend << std::endl;
      }

      b.copyTargetToSource();

      // (optional) run test
      if ((opts.do_check == dlaf::miniapp::CheckIterFreq::Last && run_index == (opts.nruns - 1)) ||
          opts.do_check == dlaf::miniapp::CheckIterFreq::All) {
        // TODO do not check element by element, but evaluate the entire matrix
        static_assert(std::is_arithmetic_v<T>, "mul/add error is valid just for arithmetic types");
        constexpr T muladd_error = 2 * std::numeric_limits<T>::epsilon();

        const T max_error = 20 * (bh.size().rows() + 1) * muladd_error;
        CHECK_MATRIX_NEAR(ref_x, bh, max_error, 0);
      }
    }
  }
};

int pika_main(pika::program_options::variables_map& vm) {
  {
    dlaf::ScopedInitializer init(vm);
    const Options opts(vm);

    dlaf::miniapp::dispatchMiniapp<triangularSolverMiniapp>(opts);
  }

  return pika::finalize();
}

int main(int argc, char** argv) {
  dlaf::comm::mpi_init mpi_initter(argc, argv, dlaf::comm::mpi_thread_level::multiple);

  // options
  using namespace pika::program_options;
  options_description desc_commandline(
      "Benchmark computation of solution for A . X = 2 . B, "
      "where A is a non-unit lower triangular matrix, and B is an m by n matrix\n\n"
      "options");
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

namespace {
/// Returns a tuple of element generators of three matrices A(m x m), B (m x n), X (m x n), for which it
/// holds op(A) X = alpha B (alpha can be any value).
///
/// The elements of op(A) (@p el_op_a) are chosen such that:
///   op(A)_ik = (i+1) / (k+.5) * exp(I*(2*i-k)) for the referenced elements
///   op(A)_ik = -9.9 otherwise,
/// where I = 0 for real types or I is the complex unit for complex types.
///
/// The elements of X (@p el_x) are computed as
///   X_kj = (k+.5) / (j+2) * exp(I*(k+j)).
/// These data are typically used to check whether the result of the equation
/// performed with any algorithm is consistent with the computed values.
///
/// Finally, the elements of B (@p el_b) should be:
/// B_ij = (Sum_k op(A)_ik * X_kj) / alpha
///      = (op(A)_ii * X_ij + (kk-1) * gamma) / alpha,
/// where gamma = (i+1) / (j+2) * exp(I*(2*i+j)),
///       kk = i+1 if op(a) is an lower triangular matrix, or
///       kk = m-i if op(a) is an lower triangular matrix.
/// Therefore
/// B_ij = (X_ij + (kk-1) * gamma) / alpha, if diag == Unit
/// B_ij = kk * gamma / alpha, otherwise.
template <typename T>
linear_system_t<T> sampleLeftTr(blas::Uplo uplo, blas::Op op, blas::Diag diag, T alpha, SizeType m) {
  static_assert(std::is_arithmetic_v<T> && !std::is_integral_v<T>,
                "it is valid just with floating point values");

  bool op_a_lower = (uplo == blas::Uplo::Lower && op == blas::Op::NoTrans) ||
                    (uplo == blas::Uplo::Upper && op != blas::Op::NoTrans);

  auto el_op_a = [op_a_lower, diag](const GlobalElementIndex& index) -> T {
    if ((op_a_lower && index.row() < index.col()) || (!op_a_lower && index.row() > index.col()) ||
        (diag == blas::Diag::Unit && index.row() == index.col()))
      return static_cast<T>(-9.9);

    const T i = index.row();
    const T k = index.col();

    return (i + static_cast<T>(1)) / (k + static_cast<T>(.5));
  };

  auto el_x = [](const GlobalElementIndex& index) -> T {
    const T k = index.row();
    const T j = index.col();

    return (k + static_cast<T>(.5)) / (j + static_cast<T>(2));
  };

  auto el_b = [m, alpha, diag, op_a_lower, el_x](const GlobalElementIndex& index) -> T {
    const dlaf::BaseType<T> kk = op_a_lower ? index.row() + 1 : m - index.row();

    const T i = index.row();
    const T j = index.col();
    const T gamma = (i + 1) / (j + 2);
    if (diag == blas::Diag::Unit)
      return ((kk - 1) * gamma + el_x(index)) / alpha;
    else
      return kk * gamma / alpha;
  };

  return {el_op_a, el_b, el_x};
}

}
