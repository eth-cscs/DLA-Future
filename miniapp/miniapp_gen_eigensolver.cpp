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

#include <pika/init.hpp>
#include <pika/program_options.hpp>
#include <pika/runtime.hpp>

#include <dlaf/auxiliary/norm.h>
#include <dlaf/common/assert.h>
#include <dlaf/common/format_short.h>
#include <dlaf/common/index2d.h>
#include <dlaf/common/range2d.h>
#include <dlaf/common/timer.h>
#include <dlaf/communication/communicator_grid.h>
#include <dlaf/communication/init.h>
#include <dlaf/eigensolver/gen_eigensolver.h>
#include <dlaf/eigensolver/internal/get_band_size.h>
#include <dlaf/init.h>
#include <dlaf/matrix/copy.h>
#include <dlaf/matrix/hdf5.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/matrix_mirror.h>
#include <dlaf/miniapp/dispatch.h>
#include <dlaf/miniapp/options.h>
#include <dlaf/miniapp/scale_eigenvectors.h>
#include <dlaf/multiplication/hermitian.h>
#include <dlaf/types.h>
#include <dlaf/util_math.h>

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

#ifdef DLAF_WITH_HDF5
using dlaf::matrix::internal::FileHDF5;
#endif

/// Check results of the eigensolver
template <typename T>
void checkGenEigensolver(CommunicatorGrid& comm_grid, blas::Uplo uplo, Matrix<const T, Device::CPU>& A,
                         Matrix<const T, Device::CPU>& B,
                         Matrix<const BaseType<T>, Device::CPU>& evalues,
                         Matrix<const T, Device::CPU>& E, const SizeType eval_idx_end);

struct Options
    : dlaf::miniapp::MiniappOptions<dlaf::miniapp::SupportReal::Yes, dlaf::miniapp::SupportComplex::Yes> {
  SizeType m;
  SizeType mb;
  SizeType eval_idx_end;
  blas::Uplo uplo;
#ifdef DLAF_WITH_HDF5
  std::filesystem::path input_file;
  std::string input_dataset_a;
  std::string input_dataset_b;
  std::filesystem::path output_file;
#endif

  Options(const pika::program_options::variables_map& vm)
      : MiniappOptions(vm), m(vm["matrix-size"].as<SizeType>()), mb(vm["block-size"].as<SizeType>()),
        uplo(dlaf::miniapp::parseUplo(vm["uplo"].as<std::string>())) {
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
      eval_idx_end = dlaf::util::internal::percent_to_index(m, percent);
    }
    else if (vm.count("eval-index-end") == 1) {
      eval_idx_end = vm["eval-index-end"].as<SizeType>();
    }
    else {
      eval_idx_end = m;
    }

    DLAF_ASSERT(eval_idx_end >= 0 && eval_idx_end <= m, eval_idx_end, m);

#ifdef DLAF_WITH_HDF5
    if (vm.count("input-file") == 1) {
      input_file = vm["input-file"].as<std::filesystem::path>();

      if (!vm["matrix-size"].defaulted()) {
        std::cerr << "Warning! "
                     "Specified matrix size will be ignored because an input file has been specified."
                  << std::endl;
      }
    }
    input_dataset_a = vm["input-dataset-a"].as<std::string>();
    input_dataset_b = vm["input-dataset-b"].as<std::string>();
    if (vm.count("output-file") == 1) {
      output_file = vm["output-file"].as<std::filesystem::path>();
    }
#endif
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

    ConstHostMatrixType matrix_a_ref = [matrix_size, block_size, &comm_grid, &opts]() {
#ifdef DLAF_WITH_HDF5
      if (!opts.input_file.empty()) {
        auto infile = FileHDF5(opts.input_file, FileHDF5::FileMode::readonly);
        if (opts.local)
          return infile.read<T>(opts.input_dataset_a, block_size);
        else
          return infile.read<T>(opts.input_dataset_a, block_size, comm_grid, {0, 0});
      }
#else
      dlaf::internal::silenceUnusedWarningFor(opts);
#endif
      using dlaf::matrix::util::set_random_hermitian;

      HostMatrixType hermitian(matrix_size, block_size, comm_grid);
      set_random_hermitian(hermitian);

      return hermitian;
    }();

    // Default capture & is needed to suppress warning of unused opts when DLAF_WITH_HDF5 is not defined.
    // [[maybe_unused]] is not supported in lambda captures
    ConstHostMatrixType matrix_b_ref = [&, matrix_size, block_size]() {
#ifdef DLAF_WITH_HDF5
      if (!opts.input_file.empty()) {
        auto infile = FileHDF5(opts.input_file, FileHDF5::FileMode::readonly);
        if (opts.local)
          return infile.read<T>(opts.input_dataset_b, block_size);
        else
          return infile.read<T>(opts.input_dataset_b, block_size, comm_grid, {0, 0});
      }
#endif
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
        using dlaf::hermitian_generalized_eigensolver;
        if (opts.local)
          return hermitian_generalized_eigensolver<backend>(opts.uplo, matrix_a->get(), matrix_b->get(),
                                                            0l, opts.eval_idx_end);
        else
          return hermitian_generalized_eigensolver<backend>(comm_grid, opts.uplo, matrix_a->get(),
                                                            matrix_b->get(), 0l, opts.eval_idx_end);
      };
      auto [eigenvalues, eigenvectors] = bench();

      // wait and barrier for all ranks
      eigenvectors.waitLocalTiles();
      comm_grid.wait_all_communicators();
      double elapsed_time = timeit.elapsed();

#ifdef DLAF_WITH_HDF5
      if (run_index == opts.nruns - 1) {
        if (!opts.output_file.empty()) {
          auto outfile = [&]() {
            if (opts.local)
              return FileHDF5(opts.output_file, FileHDF5::FileMode::readwrite);
            else
              return FileHDF5(world, opts.output_file);
          }();
          outfile.write(matrix_a_ref, opts.input_dataset_a);
          outfile.write(matrix_b_ref, opts.input_dataset_b);
          outfile.write(eigenvalues, "/evals");
          outfile.write(eigenvectors, "/evecs");
        }
      }
#endif

      matrix_a.reset();
      matrix_b.reset();

      // print benchmark results
      if (0 == world.rank() && run_index >= 0) {
        std::cout << "[" << run_index << "]" << " " << elapsed_time << "s" << " "
                  << dlaf::internal::FormatShort{opts.type} << dlaf::internal::FormatShort{opts.uplo}
                  << " " << matrix_a_host.size() << " (" << 0l << ", " << opts.eval_idx_end << ") "
                  << " " << matrix_a_host.blockSize() << " "
                  << dlaf::eigensolver::internal::getBandSize(matrix_a_host.blockSize().rows()) << " "
                  << comm_grid.size() << " " << pika::get_os_thread_count() << " " << backend
                  << std::endl;
        if (opts.csv_output) {
          // CSV formatted output with column names that can be read by pandas to simplify
          // post-processing CSVData{-version}, value_0, title_0, value_1, title_1
          std::cout << "CSVData-2, "
                    << "run, " << run_index << ", "
                    << "time, " << elapsed_time << ", "
                    << "type, " << dlaf::internal::FormatShort{opts.type}.value << ", "
                    << "uplo, " << dlaf::internal::FormatShort{opts.uplo}.value << ", "
                    << "matrixsize, " << matrix_a_host.size().rows() << ", "
                    << "blocksize, " << matrix_a_host.blockSize().rows() << ", "
                    << "bandsize, "
                    << dlaf::eigensolver::internal::getBandSize(matrix_a_host.blockSize().rows()) << ", "
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
        MatrixMirrorEvalsType eigenvalues_host(eigenvalues);
        MatrixMirrorEvectsType eigenvectors_host(eigenvectors);
        checkGenEigensolver(comm_grid, opts.uplo, matrix_a_ref, matrix_b_ref, eigenvalues_host.get(),
                            eigenvectors_host.get(), opts.eval_idx_end);
      }

      eigenvalues.waitLocalTiles();
      eigenvectors.waitLocalTiles();
    }

    comm_grid.wait_all_communicators();
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
    ("matrix-size",    value<SizeType>()   ->default_value(4096), "Matrix size")
    ("block-size",     value<SizeType>()   ->default_value( 256), "Block cyclic distribution size")
    ("eval-index-end", value<SizeType>()                        , "Index of last eigenvalue of interest/eigenvector to transform (exclusive)")
    ("percent-evals",  value<double>()                          , "Percentage of eigenvalues of interest/eigenvectors to transform")
#ifdef DLAF_WITH_HDF5
    ("input-file",    value<std::filesystem::path>()                              , "Load matrix from given HDF5 file")
    ("input-dataset-a", value<std::string>()         -> default_value("/input-a") , "Name of HDF5 dataset to load as matrix")
    ("input-dataset-b", value<std::string>()         -> default_value("/input-b") , "Name of HDF5 dataset to load as matrix")
    ("output-file",   value<std::filesystem::path>()                              , "Save eigenvectors and eigenvalues to given HDF5 file")
#endif
  ;
  // clang-format on
  dlaf::miniapp::addUploOption(desc_commandline);

  pika::init_params p;
  p.desc_cmdline = desc_commandline;
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
void checkGenEigensolver(CommunicatorGrid& comm_grid, blas::Uplo uplo, Matrix<const T, Device::CPU>& A,
                         Matrix<const T, Device::CPU>& B,
                         Matrix<const BaseType<T>, Device::CPU>& evalues,
                         Matrix<const T, Device::CPU>& E, const SizeType eval_idx_end) {
  const Index2D rank_result{0, 0};

  const auto norm_A =
      sync_wait(dlaf::auxiliary::max_norm<dlaf::Backend::MC>(comm_grid, rank_result, uplo, A));
  const auto norm_B =
      sync_wait(dlaf::auxiliary::max_norm<dlaf::Backend::MC>(comm_grid, rank_result, uplo, B));

  // 2.
  // Compute C = E D - A E
  auto spec = dlaf::matrix::util::internal::sub_matrix_spec_slice_cols(E, 0l, eval_idx_end);
  dlaf::matrix::internal::MatrixRef<const T, Device::CPU> E_ref(E, spec);
  dlaf::matrix::internal::MatrixRef<const BaseType<T>, Device::CPU> evalues_ref(
      evalues, {{0, 0}, {eval_idx_end, 1}});

  Matrix<T, Device::CPU> C(E_ref.distribution());
  Matrix<T, Device::CPU> C2(E_ref.distribution());
  dlaf::miniapp::scaleEigenvectors(evalues_ref, E_ref, C2);
  dlaf::hermitian_multiplication<Backend::MC>(comm_grid, blas::Side::Left, uplo, T{1}, B, C2, T{0}, C);
  dlaf::internal::hermitian_multiplication<Backend::MC>(comm_grid, blas::Side::Left, uplo, T{-1}, A,
                                                        E_ref, T{1}, C);

  // 3. Compute the max norm of the difference
  const auto norm_diff = sync_wait(dlaf::auxiliary::max_norm<dlaf::Backend::MC>(comm_grid, rank_result,
                                                                                blas::Uplo::General, C));

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
