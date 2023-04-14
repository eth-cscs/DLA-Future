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
#include <filesystem>
#include <iostream>
#include <limits>
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
#include <dlaf/eigensolver/tridiag_solver.h>
#include <dlaf/init.h>
#include <dlaf/matrix/copy.h>
#include <dlaf/matrix/hdf5.h>
#include <dlaf/matrix/index.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/matrix_mirror.h>
#include <dlaf/miniapp/dispatch.h>
#include <dlaf/miniapp/options.h>
#include <dlaf/types.h>

namespace {
using dlaf::Backend;
using dlaf::DefaultDevice_v;
using dlaf::Device;
using dlaf::GlobalElementIndex;
using dlaf::GlobalElementSize;
using dlaf::LocalElementSize;
using dlaf::Matrix;
using dlaf::SizeType;
using dlaf::TileElementSize;
using dlaf::comm::Communicator;
using dlaf::comm::CommunicatorGrid;
using dlaf::common::Ordering;
using dlaf::matrix::Distribution;
using dlaf::matrix::MatrixMirror;

#ifdef DLAF_WITH_HDF5
using dlaf::matrix::internal::FileHDF5;
#endif

struct Options
    : dlaf::miniapp::MiniappOptions<dlaf::miniapp::SupportReal::Yes, dlaf::miniapp::SupportComplex::Yes> {
  SizeType m;
  SizeType mb;
#ifdef DLAF_WITH_HDF5
  std::filesystem::path input_file;
  std::string input_dataset;
#endif

  Options(const pika::program_options::variables_map& vm)
      : MiniappOptions(vm), m(vm["matrix-size"].as<SizeType>()), mb(vm["block-size"].as<SizeType>()) {
    DLAF_ASSERT(m > 0, m);
    DLAF_ASSERT(mb > 0, mb);

#ifdef DLAF_WITH_HDF5
    if (vm.count("input-file") == 1) {
      input_file = vm["input-file"].as<std::filesystem::path>();

      if (!vm["matrix-size"].defaulted()) {
        std::cerr << "Warning! "
                     "Specified matrix size will be ignored because an input file has been specified."
                  << std::endl;
      }
    }
    input_dataset = vm["input-dataset"].as<std::string>();
#endif

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

struct TridiagSolverMiniapp {
  template <Backend backend, typename T>
  static void run(const Options& opts) {
    using RealT = dlaf::BaseType<T>;
    Communicator world(MPI_COMM_WORLD);
    CommunicatorGrid comm_grid(world, opts.grid_rows, opts.grid_cols, Ordering::ColumnMajor);

    Matrix<const RealT, Device::CPU> tridiag_ref = [&opts]() {
#ifdef DLAF_WITH_HDF5
      if (!opts.input_file.empty()) {
        auto infile = FileHDF5(opts.input_file, FileHDF5::FileMode::readonly);
        Matrix<RealT, Device::CPU> tridiag = infile.read<RealT>(opts.input_dataset, {opts.mb, 2});
        return tridiag;
      }
#endif
      const Distribution dist_trd(LocalElementSize(opts.m, 2), TileElementSize(opts.mb, 2));
      Matrix<RealT, Device::CPU> tridiag(dist_trd);
      dlaf::matrix::util::set_random(tridiag);
      return tridiag;
    }();

    const Distribution dist_evals(LocalElementSize(tridiag_ref.size().rows(), 1),
                                  TileElementSize(opts.mb, 1));
    const Distribution dist_evecs(
        GlobalElementSize(tridiag_ref.size().rows(), tridiag_ref.size().rows()),
        TileElementSize(opts.mb, opts.mb), comm_grid.size(), comm_grid.rank(), {0, 0});

    Matrix<RealT, Device::CPU> tridiag(tridiag_ref.distribution());
    Matrix<RealT, Device::CPU> evals(dist_evals);
    Matrix<T, Device::CPU> evecs(dist_evecs);

    for (int64_t run_index = -opts.nwarmups; run_index < opts.nruns; ++run_index) {
      if (0 == world.rank() && run_index >= 0)
        std::cout << "[" << run_index << "]" << std::endl;

      copy(tridiag_ref, tridiag);

      double elapsed_time;
      {
        MatrixMirror<RealT, DefaultDevice_v<backend>, Device::CPU> evals_mirror(evals);
        MatrixMirror<T, DefaultDevice_v<backend>, Device::CPU> evecs_mirror(evecs);

        // Wait for matrix to be copied to GPU (if necessary)
        tridiag.waitLocalTiles();
        evals_mirror.get().waitLocalTiles();
        evecs_mirror.get().waitLocalTiles();
        DLAF_MPI_CHECK_ERROR(MPI_Barrier(world));

        dlaf::common::Timer<> timeit;
        using dlaf::eigensolver::internal::tridiagonal_eigensolver;
        if (opts.local)
          tridiagonal_eigensolver<backend>(tridiag, evals_mirror.get(), evecs_mirror.get());
        else
          tridiagonal_eigensolver<backend>(comm_grid, tridiag, evals_mirror.get(), evecs_mirror.get());

        // wait and barrier for all ranks
        tridiag.waitLocalTiles();
        evals_mirror.get().waitLocalTiles();
        evecs_mirror.get().waitLocalTiles();
        DLAF_MPI_CHECK_ERROR(MPI_Barrier(world));
        elapsed_time = timeit.elapsed();
      }

      // print benchmark results
      if (0 == world.rank() && run_index >= 0) {
        std::cout << "[" << run_index << "]"
                  << " " << elapsed_time << "s"
                  << " " << dlaf::internal::FormatShort{opts.type} << " " << tridiag.size() << " "
                  << tridiag.blockSize() << " " << comm_grid.size() << " " << pika::get_os_thread_count()
                  << " " << backend << std::endl;
        if (opts.csv_output) {
          // CSV formatted output with column names that can be read by pandas to simplify
          // post-processing CSVData{-version}, value_0, title_0, value_1, title_1
          std::cout << "CSVData-2, "
                    << "run, " << run_index << ", "
                    << "time, " << elapsed_time << ", "
                    << "type, " << dlaf::internal::FormatShort{opts.type}.value << ", "
                    << "matrixsize, " << tridiag.size().rows() << ", "
                    << "blocksize, " << tridiag.blockSize().rows() << ", "
                    << "comm_rows, " << comm_grid.size().rows() << ", "
                    << "comm_cols, " << comm_grid.size().cols() << ", "
                    << "threads, " << pika::get_os_thread_count() << ", "
                    << "backend, " << backend << ", " << opts.info << std::endl;
        }
      }

      // (optional) run test
      if ((opts.do_check == dlaf::miniapp::CheckIterFreq::Last && run_index == (opts.nruns - 1)) ||
          opts.do_check == dlaf::miniapp::CheckIterFreq::All) {
        // TODO implement check
        DLAF_UNIMPLEMENTED("Check");
      }
    }
  }
};

int pika_main(pika::program_options::variables_map& vm) {
  pika::scoped_finalize pika_finalizer;
  dlaf::ScopedInitializer init(vm);

  const Options opts(vm);
  dlaf::miniapp::dispatchMiniapp<TridiagSolverMiniapp>(opts);

  return EXIT_SUCCESS;
}

int main(int argc, char** argv) {
  // Init MPI
  dlaf::comm::mpi_init mpi_initter(argc, argv);

  // options
  using namespace pika::program_options;
  options_description desc_commandline("Usage: miniapp_tridiag_solver [options]");
  desc_commandline.add(dlaf::miniapp::getMiniappOptionsDescription());
  desc_commandline.add(dlaf::getOptionsDescription());

  // clang-format off
  desc_commandline.add_options()
    ("matrix-size",  value<SizeType>()   ->default_value(4096), "Matrix size")
    ("block-size",   value<SizeType>()   ->default_value( 256), "Block cyclic distribution size")
#ifdef DLAF_WITH_HDF5
    ("input-file",    value<std::filesystem::path>()                             , "Load matrix from given HDF5 file")
    ("input-dataset", value<std::string>()           ->default_value("/tridiag") , "Name of HDF5 dataset to load as matrix")
#endif
  ;
  // clang-format on
  dlaf::miniapp::addUploOption(desc_commandline);

  pika::init_params p;
  p.desc_cmdline = desc_commandline;
  p.rp_callback = dlaf::initResourcePartitionerHandler;
  return pika::init(pika_main, argc, argv, p);
}
