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
#include <limits>
#include <type_traits>

#include <mpi.h>
#include <hpx/init.hpp>
#include <hpx/program_options.hpp>
#include <hpx/runtime.hpp>

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

namespace {

using dlaf::Backend;
using dlaf::Device;
using dlaf::GlobalElementIndex;
using dlaf::GlobalElementSize;
using dlaf::SizeType;
using dlaf::TileElementSize;
using dlaf::comm::Communicator;
using dlaf::comm::CommunicatorGrid;
using dlaf::common::Ordering;
using dlaf::comm::MPIMech;

using T = double;

struct options_t {
  SizeType m;
  SizeType n;
  SizeType mb;
  SizeType nb;
  int grid_rows;
  int grid_cols;
  int64_t nruns;
  int64_t nwarmups;
  bool do_check;
};

options_t check_options(hpx::program_options::variables_map& vm);

using matrix_values_t = std::function<T(const GlobalElementIndex&)>;
using linear_system_t = std::tuple<matrix_values_t, matrix_values_t, matrix_values_t>;  // A, B, X
linear_system_t sampleLeftTr(blas::Uplo uplo, blas::Op op, blas::Diag diag, T alpha, SizeType m);
}

int hpx_main(hpx::program_options::variables_map& vm) {
  dlaf::initialize(vm);
  {
    options_t opts = check_options(vm);

    Communicator world(MPI_COMM_WORLD);
    CommunicatorGrid comm_grid(world, opts.grid_rows, opts.grid_cols, Ordering::ColumnMajor);

    // Allocate memory for the matrices
    dlaf::matrix::Matrix<T, Device::CPU> ah(GlobalElementSize{opts.m, opts.m},
                                            TileElementSize{opts.mb, opts.mb}, comm_grid);
    dlaf::matrix::Matrix<T, Device::CPU> bh(GlobalElementSize{opts.m, opts.n},
                                            TileElementSize{opts.mb, opts.nb}, comm_grid);

    dlaf::matrix::MatrixMirror<T, Device::Default, Device::CPU> a(ah);
    dlaf::matrix::MatrixMirror<T, Device::Default, Device::CPU> b(bh);

    auto sync_barrier = [&]() {
      a.get().waitLocalTiles();
      b.get().waitLocalTiles();
      DLAF_MPI_CALL(MPI_Barrier(world));
    };

    const auto side = blas::Side::Left;
    const auto uplo = blas::Uplo::Lower;
    const auto op = blas::Op::NoTrans;
    const auto diag = blas::Diag::NonUnit;
    const T alpha = 2.0;

    double m = ah.size().rows();
    double n = bh.size().cols();
    auto add_mul = n * m * m / 2;
    const double total_ops = dlaf::total_ops<T>(add_mul, add_mul);

    matrix_values_t ref_a, ref_b, ref_x;
    std::tie(ref_a, ref_b, ref_x) = ::sampleLeftTr(uplo, op, diag, alpha, ah.size().rows());

    for (int64_t run_index = -opts.nwarmups; run_index < opts.nruns; ++run_index) {
      if (0 == world.rank() && run_index >= 0)
        std::cout << "[" << run_index << "]" << std::endl;

      // setup matrix A and b
      using dlaf::matrix::util::set;
      set(ah, ref_a);
      set(bh, ref_b);
      a.copySourceToTarget();
      b.copySourceToTarget();

      sync_barrier();

      dlaf::common::Timer<> timeit;
      dlaf::solver::triangular<Backend::Default, Device::Default, T>(comm_grid, side, uplo, op, diag,
                                                                     alpha, a.get(), b.get());

      sync_barrier();

      // benchmark results
      if (0 == world.rank() && run_index >= 0) {
        auto elapsed_time = timeit.elapsed();
        double gigaflops = total_ops / elapsed_time / 1e9;

        std::cout << "[" << run_index << "]"
                  << " " << elapsed_time << "s"
                  << " " << gigaflops << "GFlop/s"
                  << " " << bh.size() << " " << bh.blockSize() << " " << comm_grid.size() << " "
                  << hpx::get_os_thread_count() << std::endl;
      }

      b.copyTargetToSource();

      // (optional) run test
      if (opts.do_check) {
        // TODO do not check element by element, but evaluate the entire matrix

        static_assert(std::is_arithmetic<T>::value, "mul/add error is valid just for arithmetic types");
        constexpr T muladd_error = 2 * std::numeric_limits<T>::epsilon();

        const T max_error = 20 * (bh.size().rows() + 1) * muladd_error;
        CHECK_MATRIX_NEAR(ref_x, bh, max_error, 0);
      }
    }
  }

  dlaf::finalize();

  return hpx::finalize();
}

int main(int argc, char** argv) {
  dlaf::comm::mpi_init mpi_initter(argc, argv, dlaf::comm::mpi_thread_level::multiple);

  // options
  using namespace hpx::program_options;
  options_description desc_commandline(
      "Benchmark computation of solution for A . X = 2 . B, "
      "where A is a non-unit lower triangular matrix, and B is an m by n matrix\n\n"
      "options");

  // clang-format off
  desc_commandline.add_options()
    ("m",             value<SizeType>()  ->default_value(4096),       "Matrix b rows")
    ("n",             value<SizeType>()  ->default_value(512),        "Matrix b columns")
    ("mb",            value<SizeType>()  ->default_value(256),        "Matrix b block rows")
    ("nb",            value<SizeType>()  ->default_value(512),        "Matrix b block columns")
    ("grid-rows",     value<int>()       ->default_value(1),          "Number of row processes in the 2D communicator.")
    ("grid-cols",     value<int>()       ->default_value(1),          "Number of column processes in the 2D communicator.")
    ("nruns",         value<int64_t>()   ->default_value(1),          "Number of runs to compute the cholesky")
    ("nwarmups",      value<int64_t>()   ->default_value(1),          "Number of warmup runs")
    ("check-result",  bool_switch()      ->default_value(false),      "Check the triangular system solution (for each run)")
  ;
  // clang-format on

  desc_commandline.add(dlaf::getOptionsDescription());

  hpx::init_params p;
  p.desc_cmdline = desc_commandline;
  p.rp_callback = dlaf::initResourcePartitionerHandler;
  return hpx::init(argc, argv, p);
}

namespace {

options_t check_options(hpx::program_options::variables_map& vm) {
  // clang-format off
  options_t opts = {
    vm["m"].as<SizeType>(),     vm["n"].as<SizeType>(),
    vm["mb"].as<SizeType>(),    vm["nb"].as<SizeType>(),
    vm["grid-rows"].as<int>(),  vm["grid-cols"].as<int>(),

    vm["nruns"].as<int64_t>(),
    vm["nwarmups"].as<int64_t>(),
    vm["check-result"].as<bool>(),
  };
  // clang-format on

  DLAF_ASSERT(opts.m > 0 && opts.n > 0, opts.m, opts.n);
  DLAF_ASSERT(opts.mb > 0 && opts.nb > 0, opts.mb, opts.nb);
  DLAF_ASSERT(opts.grid_rows > 0 && opts.grid_cols > 0, opts.grid_rows, opts.grid_cols);
  DLAF_ASSERT(opts.nruns > 0, opts.nruns);
  DLAF_ASSERT(opts.nwarmups >= 0, opts.nwarmups);

  return opts;
}

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
linear_system_t sampleLeftTr(blas::Uplo uplo, blas::Op op, blas::Diag diag, T alpha, SizeType m) {
  static_assert(std::is_arithmetic<T>::value && !std::is_integral<T>::value,
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
