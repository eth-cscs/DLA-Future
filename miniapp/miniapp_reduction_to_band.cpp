//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
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

#include <dlaf/common/format_short.h>
#include <dlaf/common/index2d.h>
#include <dlaf/common/range2d.h>
#include <dlaf/common/timer.h>
#include <dlaf/communication/communicator_grid.h>
#include <dlaf/communication/init.h>
#include <dlaf/eigensolver/reduction_to_band.h>
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
using dlaf::Device;
using dlaf::SizeType;

#ifdef DLAF_WITH_HDF5
using dlaf::matrix::internal::FileHDF5;
#endif

struct Options
    : dlaf::miniapp::MiniappOptions<dlaf::miniapp::SupportReal::Yes, dlaf::miniapp::SupportComplex::Yes> {
  SizeType m;
  SizeType mb;
  SizeType b;
#ifdef DLAF_WITH_HDF5
  std::filesystem::path input_file;
  std::string input_dataset;
  std::filesystem::path output_file;
#endif

  Options(const pika::program_options::variables_map& vm)
      : MiniappOptions(vm), m(vm["matrix-size"].as<SizeType>()), mb(vm["block-size"].as<SizeType>()),
        b(vm["band-size"].as<SizeType>()) {
    DLAF_ASSERT(m > 0, m);
    DLAF_ASSERT(mb > 0, mb);

    if (b < 0)
      b = mb;

    DLAF_ASSERT(b > 0 && (mb % b == 0), b, mb);

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
    if (vm.count("output-file") == 1) {
      output_file = vm["output-file"].as<std::filesystem::path>();
    }
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

struct reductionToBandMiniapp {
  template <dlaf::Backend backend, typename T>
  static void run(const Options& opts) {
    using namespace dlaf;
    using dlaf::SizeType;
    using dlaf::comm::Communicator;
    using dlaf::comm::CommunicatorGrid;
    using MatrixMirrorType = matrix::MatrixMirror<T, DefaultDevice_v<backend>, Device::CPU>;
    using HostMatrixType = Matrix<T, Device::CPU>;
    using ConstMatrixType = Matrix<const T, Device::CPU>;

    Communicator world(MPI_COMM_WORLD);
    CommunicatorGrid comm_grid(world, opts.grid_rows, opts.grid_cols, common::Ordering::ColumnMajor);

    ConstMatrixType matrix_ref = [&comm_grid, &opts]() {
#ifdef DLAF_WITH_HDF5
      if (!opts.input_file.empty()) {
        auto infile = FileHDF5(opts.input_file, FileHDF5::FileMode::readonly);
        if (opts.local)
          return infile.read<T>(opts.input_dataset, {opts.mb, opts.mb});
        else
          return infile.read<T>(opts.input_dataset, {opts.mb, opts.mb}, comm_grid, {0, 0});
      }
#endif
      using dlaf::matrix::util::set_random_hermitian;

      HostMatrixType hermitian(GlobalElementSize(opts.m, opts.m), TileElementSize(opts.mb, opts.mb),
                               comm_grid);
      set_random_hermitian(hermitian);

      return hermitian;
    }();

    auto matrix_size = matrix_ref.size();
    auto block_size = matrix_ref.blockSize();

    for (int64_t run_index = -opts.nwarmups; run_index < opts.nruns; ++run_index) {
      if (0 == world.rank() && run_index >= 0)
        std::cout << "[" << run_index << "]" << std::endl;

      HostMatrixType matrix_host(matrix_size, block_size, comm_grid);
      copy(matrix_ref, matrix_host);

      double elapsed_time;
      {
        MatrixMirrorType matrix_mirror(matrix_host);

        Matrix<T, DefaultDevice_v<backend>>& matrix = matrix_mirror.get();

        // wait all setup tasks before starting benchmark
        matrix_host.waitLocalTiles();
        DLAF_MPI_CHECK_ERROR(MPI_Barrier(world));

        dlaf::common::Timer<> timeit;
        auto bench = [&]() {
          if (opts.local)
            return dlaf::eigensolver::internal::reduction_to_band<backend>(matrix, opts.b);
          else
            return dlaf::eigensolver::internal::reduction_to_band<backend>(comm_grid, matrix, opts.b);
        };
        auto taus = bench();

        // wait and barrier for all ranks
        matrix.waitLocalTiles();
        taus.waitLocalTiles();
        comm_grid.wait_all_communicators();

        elapsed_time = timeit.elapsed();
      }

      double gigaflops = std::numeric_limits<double>::quiet_NaN();
      {
        double n = matrix_host.size().rows();
        double b = matrix_host.blockSize().rows();
        auto add_mul = 2. / 3. * n * n * n - n * n * b;
        gigaflops = dlaf::total_ops<T>(add_mul, add_mul) / elapsed_time / 1e9;
      }

#ifdef DLAF_WITH_HDF5
      if (run_index == opts.nruns - 1) {
        if (!opts.output_file.empty()) {
          auto outfile = [&]() {
            if (opts.local)
              return FileHDF5(opts.output_file, FileHDF5::FileMode::readwrite);
            else
              return FileHDF5(world, opts.output_file);
          }();
          outfile.write(matrix_ref, opts.input_dataset);
          outfile.write(matrix_host, "/band");
        }
      }
#endif

      // print benchmark results
      if (0 == world.rank() && run_index >= 0) {
        std::cout << "[" << run_index << "]"
                  << " " << elapsed_time << "s"
                  << " " << gigaflops << "GFlop/s"
                  << " " << dlaf::internal::FormatShort{opts.type} << " " << matrix_host.size() << " "
                  << matrix_host.blockSize() << " " << opts.b << " " << comm_grid.size() << " "
                  << pika::get_os_thread_count() << " " << backend << std::endl;
        if (opts.csv_output) {
          // CSV formatted output with column names that can be read by pandas to simplify
          // post-processing CSVData{-version}, value_0, title_0, value_1, title_1
          std::cout << "CSVData-2, "
                    << "run, " << run_index << ", "
                    << "time, " << elapsed_time << ", "
                    << "GFlops, " << gigaflops << ", "
                    << "type, " << dlaf::internal::FormatShort{opts.type}.value << ", "
                    << "matrixsize, " << matrix_host.size().rows() << ", "
                    << "blocksize, " << matrix_host.blockSize().rows() << ", "
                    << "band_size, " << opts.b << ", "
                    << "comm_rows, " << comm_grid.size().rows() << ", "
                    << "comm_cols, " << comm_grid.size().cols() << ", "
                    << "threads, " << pika::get_os_thread_count() << ", "
                    << "backend, " << backend << ", " << opts.info << std::endl;
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
  dlaf::miniapp::dispatchMiniapp<reductionToBandMiniapp>(opts);

  return EXIT_SUCCESS;
}

int main(int argc, char** argv) {
  using dlaf::SizeType;

  dlaf::comm::mpi_init mpi_initter(argc, argv);

  // options
  using namespace pika::program_options;
  options_description desc_commandline("Usage: miniapp_reduction_to_band [options]");
  desc_commandline.add(dlaf::miniapp::getMiniappOptionsDescription());
  desc_commandline.add(dlaf::getOptionsDescription());

  // clang-format off
  desc_commandline.add_options()
    ("matrix-size", value<SizeType>()   ->default_value(4096), "Matrix rows")
    ("block-size",  value<SizeType>()   ->default_value( 256), "Block cyclic distribution size")
    ("band-size",   value<SizeType>()   ->default_value(  -1), "Band size (a negative value implies band-size=block-size")
#ifdef DLAF_WITH_HDF5
    ("input-file",    value<std::filesystem::path>()                 , "Load matrix from given HDF5 file")
    ("output-file",   value<std::filesystem::path>()                 , "Save band matrix to given HDF5 file")
    ("input-dataset", value<std::string>() ->default_value("/input") , "Name of HDF5 dataset to load as matrix")
#endif
  ;
  // clang-format on

  pika::init_params p;
  p.desc_cmdline = desc_commandline;
  return pika::init(pika_main, argc, argv, p);
}
