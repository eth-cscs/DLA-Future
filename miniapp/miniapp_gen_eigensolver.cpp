//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <cstdlib>
#include <iostream>
#include <limits>

#include <pika/init.hpp>
#include <pika/program_options.hpp>
#include <pika/runtime.hpp>

#include "dlaf/auxiliary/norm.h"
#include "dlaf/common/format_short.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/range2d.h"
#include "dlaf/common/timer.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/init.h"
#include "dlaf/eigensolver/gen_eigensolver.h"
#include "dlaf/eigensolver/get_band_size.h"
#include "dlaf/init.h"
#include "dlaf/matrix/copy.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/matrix_mirror.h"
#include "dlaf/miniapp/dispatch.h"
#include "dlaf/miniapp/options.h"
#include "dlaf/miniapp/scale_eigenvectors.h"
#include "dlaf/multiplication/hermitian.h"
#include "dlaf/types.h"

namespace {
using dlaf::Backend;
using dlaf::BaseType;
using dlaf::DefaultDevice_v;
using dlaf::Device;
using dlaf::GlobalElementSize;
using dlaf::Matrix;
using dlaf::SizeType;
using dlaf::TileElementSize;
using dlaf::comm::Communicator;
using dlaf::comm::CommunicatorGrid;
using dlaf::common::Ordering;
using dlaf::matrix::MatrixMirror;
using pika::this_thread::experimental::sync_wait;

/// Check results of the eigensolver
template <typename T>
void checkGenEigensolver(CommunicatorGrid comm_grid, blas::Uplo uplo, Matrix<const T, Device::CPU>& A,
                         Matrix<const T, Device::CPU>& B,
                         Matrix<const BaseType<T>, Device::CPU>& evalues,
                         Matrix<const T, Device::CPU>& E);

struct Options
    : dlaf::miniapp::MiniappOptions<dlaf::miniapp::SupportReal::Yes, dlaf::miniapp::SupportComplex::Yes> {
  SizeType m;
  SizeType mb;
  blas::Uplo uplo;

  Options(const pika::program_options::variables_map& vm)
      : MiniappOptions(vm), m(vm["matrix-size"].as<SizeType>()), mb(vm["block-size"].as<SizeType>()),
        uplo(dlaf::miniapp::parseUplo(vm["uplo"].as<std::string>())) {
    DLAF_ASSERT(m > 0, m);
    DLAF_ASSERT(mb > 0, mb);
  }

  Options(Options&&) = default;
  Options(const Options&) = default;
  Options& operator=(Options&&) = default;
  Options& operator=(const Options&) = default;
};
}

struct GenEigensolverMiniapp {
  template <Backend backend, typename T>
  static void run(const Options& opts) {
    using MatrixMirrorType = MatrixMirror<T, DefaultDevice_v<backend>, Device::CPU>;
    using MatrixMirrorEvalsType = MatrixMirror<const BaseType<T>, Device::CPU, DefaultDevice_v<backend>>;
    using MatrixMirrorEvectsType = MatrixMirror<const T, Device::CPU, DefaultDevice_v<backend>>;
    using HostMatrixType = Matrix<T, Device::CPU>;
    using ConstHostMatrixType = Matrix<const T, Device::CPU>;

    Communicator world(MPI_COMM_WORLD);
    CommunicatorGrid comm_grid(world, opts.grid_rows, opts.grid_cols, Ordering::ColumnMajor);

    // Allocate memory for the matrix
    GlobalElementSize matrix_size(opts.m, opts.m);
    TileElementSize block_size(opts.mb, opts.mb);

    ConstHostMatrixType matrix_a_ref = [matrix_size, block_size, comm_grid]() {
      using dlaf::matrix::util::set_random_hermitian;

      HostMatrixType hermitian(matrix_size, block_size, comm_grid);
      set_random_hermitian(hermitian);

      return hermitian;
    }();

    ConstHostMatrixType matrix_b_ref = [matrix_size, block_size, comm_grid]() {
      using dlaf::matrix::util::set_random_hermitian_positive_definite;

      HostMatrixType triangular(matrix_size, block_size, comm_grid);
      set_random_hermitian_positive_definite(triangular);

      return triangular;
    }();

    for (int64_t run_index = -opts.nwarmups; run_index < opts.nruns; ++run_index) {
      if (0 == world.rank() && run_index >= 0)
        std::cout << "[" << run_index << "]" << std::endl;

      HostMatrixType matrix_a_host(matrix_size, block_size, comm_grid);
      HostMatrixType matrix_b_host(matrix_size, block_size, comm_grid);
      copy(matrix_a_ref, matrix_a_host);
      copy(matrix_b_ref, matrix_b_host);

      auto matrix_a = std::make_unique<MatrixMirrorType>(matrix_a_host);
      auto matrix_b = std::make_unique<MatrixMirrorType>(matrix_b_host);

      // Wait all setup tasks and (if necessary) for matrix to be copied to GPU.
      matrix_a->get().waitLocalTiles();
      matrix_b->get().waitLocalTiles();
      DLAF_MPI_CHECK_ERROR(MPI_Barrier(world));

      dlaf::common::Timer<> timeit;
      auto bench = [&]() {
        if (opts.local)
          return dlaf::eigensolver::genEigensolver<backend>(opts.uplo, matrix_a->get(), matrix_b->get());
        else
          return dlaf::eigensolver::genEigensolver<backend>(comm_grid, opts.uplo, matrix_a->get(),
                                                            matrix_b->get());
      };
      auto [eigenvalues, eigenvectors] = bench();

      // wait and barrier for all ranks
      eigenvectors.waitLocalTiles();
      DLAF_MPI_CHECK_ERROR(MPI_Barrier(world));
      double elapsed_time = timeit.elapsed();

      matrix_a.reset();
      matrix_b.reset();

      // print benchmark results
      if (0 == world.rank() && run_index >= 0)
        std::cout << "[" << run_index << "]"
                  << " " << elapsed_time << "s"
                  << " " << dlaf::internal::FormatShort{opts.type}
                  << dlaf::internal::FormatShort{opts.uplo} << " " << matrix_a_host.size() << " "
                  << matrix_a_host.blockSize() << " "
                  << dlaf::eigensolver::internal::getBandSize(matrix_a_host.blockSize().rows()) << " "
                  << comm_grid.size() << " " << pika::get_os_thread_count() << " " << backend
                  << std::endl;

      // (optional) run test
      if ((opts.do_check == dlaf::miniapp::CheckIterFreq::Last && run_index == (opts.nruns - 1)) ||
          opts.do_check == dlaf::miniapp::CheckIterFreq::All) {
        MatrixMirrorEvalsType eigenvalues_host(eigenvalues);
        MatrixMirrorEvectsType eigenvectors_host(eigenvectors);
        checkGenEigensolver(comm_grid, opts.uplo, matrix_a_ref, matrix_b_ref, eigenvalues_host.get(),
                            eigenvectors_host.get());
      }
    }
  }
};

int pika_main(pika::program_options::variables_map& vm) {
  pika::scoped_finalize pika_finalizer;
  dlaf::ScopedInitializer init(vm);

  const Options opts(vm);
  dlaf::miniapp::dispatchMiniapp<GenEigensolverMiniapp>(opts);

  return EXIT_SUCCESS;
}

int main(int argc, char** argv) {
  // Init MPI
  dlaf::comm::mpi_init mpi_initter(argc, argv);

  // options
  using namespace pika::program_options;
  options_description desc_commandline("Usage: miniapp_gen_eigensolver [options]");
  desc_commandline.add(dlaf::miniapp::getMiniappOptionsDescription());
  desc_commandline.add(dlaf::getOptionsDescription());

  // clang-format off
  desc_commandline.add_options()
    ("matrix-size",  value<SizeType>()   ->default_value(4096), "Matrix size")
    ("block-size",   value<SizeType>()   ->default_value( 256), "Block cyclic distribution size")
  ;
  // clang-format on
  dlaf::miniapp::addUploOption(desc_commandline);

  pika::init_params p;
  p.desc_cmdline = desc_commandline;
  p.rp_callback = dlaf::initResourcePartitionerHandler;
  return pika::init(pika_main, argc, argv, p);
}

namespace {
using dlaf::Coord;
using dlaf::GlobalElementIndex;
using dlaf::GlobalTileIndex;
using dlaf::TileElementIndex;
using dlaf::comm::Index2D;
using dlaf::matrix::Tile;

/// Procedure to evaluate the result of the Eigensolver
///
/// 1. Check the value of | B E D - A E | / (| A | | B |)
///
/// Prints a message with the ratio and a note about the error:
/// "":        check ok
/// "ERROR":   error is high, there is an error in the results
/// "WARNING": error is slightly high, there can be an error in the result
template <typename T>
void checkGenEigensolver(CommunicatorGrid comm_grid, blas::Uplo uplo, Matrix<const T, Device::CPU>& A,
                         Matrix<const T, Device::CPU>& B,
                         Matrix<const BaseType<T>, Device::CPU>& evalues,
                         Matrix<const T, Device::CPU>& E) {
  const Index2D rank_result{0, 0};

  // 1. Compute the norm of the original matrix in A (largest eigenvalue) and in B
  const GlobalElementIndex last_ev(evalues.size().rows() - 1, 0);
  const GlobalTileIndex last_ev_tile = evalues.distribution().globalTileIndex(last_ev);
  const TileElementIndex last_ev_el_tile = evalues.distribution().tileElementIndex(last_ev);
  const auto norm_A = std::max(std::norm(sync_wait(evalues.read(GlobalTileIndex{0, 0})).get()({0, 0})),
                               std::norm(sync_wait(evalues.read(last_ev_tile)).get()(last_ev_el_tile)));
  const auto norm_B =
      dlaf::auxiliary::norm<dlaf::Backend::MC>(comm_grid, rank_result, lapack::Norm::Max, uplo, B);

  // 2.
  // Compute C = E D - A E
  Matrix<T, Device::CPU> C(E.distribution());
  Matrix<T, Device::CPU> C2(E.distribution());
  dlaf::miniapp::scaleEigenvectors(evalues, E, C2);
  dlaf::multiplication::hermitian<Backend::MC>(comm_grid, blas::Side::Left, uplo, T{1}, B, C2, T{0}, C);
  dlaf::multiplication::hermitian<Backend::MC>(comm_grid, blas::Side::Left, uplo, T{-1}, A, E, T{1}, C);

  // 3. Compute the max norm of the difference
  const auto norm_diff =
      dlaf::auxiliary::norm<dlaf::Backend::MC>(comm_grid, rank_result, lapack::Norm::Max,
                                               blas::Uplo::General, C);

  // 4.
  // Evaluation of correctness is done just by the master rank
  if (comm_grid.rank() != rank_result)
    return;

  constexpr auto eps = std::numeric_limits<dlaf::BaseType<T>>::epsilon();
  const auto n = A.size().rows();

  const auto diff_ratio = norm_diff / norm_A / norm_B;

  if (diff_ratio > 100 * eps * n)
    std::cout << "ERROR: ";
  else if (diff_ratio > eps * n)
    std::cout << "Warning: ";

  std::cout << "Max Diff / Max A / Max B: " << diff_ratio << std::endl;
}
}
