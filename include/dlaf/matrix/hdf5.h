//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#ifdef DLAF_WITH_HDF5

#include <complex>
#include <cstdint>
#include <string>

#include <H5Cpp.h>
#include <mpi.h>

#include "dlaf/common/assert.h"
#include "dlaf/common/range2d.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/matrix/index.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/matrix_mirror.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

namespace dlaf::matrix {

namespace internal {

template <class T>
struct hdf5_datatype {
  static const H5::PredType& type;
  static constexpr std::size_t dims = 1;
};

template <class T>
struct hdf5_datatype<std::complex<T>> {
  static const H5::PredType& type;
  static constexpr std::size_t dims = 2;
};

template <class T>
const H5::PredType& hdf5_datatype<std::complex<T>>::type = hdf5_datatype<T>::type;

template <class T>
struct hdf5_datatype<const T> : public hdf5_datatype<T> {};

template <class T, Device D>
H5::DataSpace mapFileToMemory(const GlobalElementIndex tl, const matrix::Tile<const T, D>& tile,
                              const H5::DataSpace& file) {
  const hsize_t file_counts[3] = {
      to_sizet(tile.size().cols()),
      to_sizet(tile.size().rows()),
      internal::hdf5_datatype<T>::dims,
  };
  const hsize_t file_offsets[3] = {
      to_sizet(tl.col()),
      to_sizet(tl.row()),
      0,
  };
  file.selectHyperslab(H5S_SELECT_SET, file_counts, file_offsets);

  const hsize_t memory_dims[3] = {
      to_sizet(tile.size().cols()),
      to_sizet(tile.ld()),
      internal::hdf5_datatype<T>::dims,
  };
  H5::DataSpace memory(3, memory_dims);

  const hsize_t memory_counts[3] = {
      to_sizet(tile.size().cols()),
      to_sizet(tile.size().rows()),
      internal::hdf5_datatype<T>::dims,
  };
  const hsize_t memory_offsets[3] = {0, 0, 0};
  memory.selectHyperslab(H5S_SELECT_SET, memory_counts, memory_offsets);

  return memory;
}

template <class T>
void from_dataset(const H5::DataSet& dataset, dlaf::Matrix<T, Device::CPU>& matrix) {
  namespace tt = pika::this_thread::experimental;

  const H5::DataSpace& dataspace_file = dataset.getSpace();

  const auto& dist = matrix.distribution();
  for (const auto ij : common::iterate_range2d(dist.localNrTiles())) {
    auto tile = tt::sync_wait(matrix.readwrite(ij));

    const GlobalElementIndex topleft = dist.globalElementIndex(dist.globalTileIndex(ij), {0, 0});
    const H5::DataSpace dataspace_mem = mapFileToMemory(topleft, tile, dataspace_file);

    dataset.read(tile.ptr(), internal::hdf5_datatype<T>::type, dataspace_mem, dataspace_file);
  }
}

template <class T>
void to_dataset(dlaf::Matrix<const T, Device::CPU>& mat, const H5::DataSet& dataset) {
  namespace tt = pika::this_thread::experimental;

  const auto& dist = mat.distribution();
  const H5::DataSpace& dataspace_file = dataset.getSpace();

  for (const auto ij : dlaf::common::iterate_range2d(dist.localNrTiles())) {
    auto tile_holder = tt::sync_wait(mat.read(ij));
    const auto& tile = tile_holder.get();

    const GlobalElementIndex topleft = dist.globalElementIndex(dist.globalTileIndex(ij), {0, 0});
    const H5::DataSpace dataspace_mem = mapFileToMemory(topleft, tile, dataspace_file);

    dataset.write(tile.ptr(), internal::hdf5_datatype<T>::type, dataspace_mem, dataspace_file);
  }
}

}

enum class HDF5_FILE_MODE {
  READONLY,
  READWRITE,
};

class FileHDF5 final {
public:
  /// Create/open a local file.
  ///
  /// Depending on @p mode
  /// - READONLY, the file will be opened (if should already exist)
  /// - READWRITE, the file will be created (if should not exist)
  ///
  /// @pre @p filepath should not exist if mode == READWRITE
  /// @pre @p filepath should exist if mode == READONLY
  /// @post local file that will not support parallel-write
  FileHDF5(const std::string& filepath, const HDF5_FILE_MODE& mode = HDF5_FILE_MODE::READONLY) {
    file_ = H5::H5File(filepath, mode2flags(mode));
  }

  /// Create a file that supports writing in parallel from different ranks.
  ///
  /// @pre @p filepath should not exist
  /// @post file created will support parallel-write
  FileHDF5(comm::Communicator comm, const std::string& filepath) {
    H5::FileAccPropList fapl;
    DLAF_ASSERT(H5Pset_fapl_mpio(fapl.getId(), comm, MPI_INFO_NULL) >= 0, "Problem setting up MPI-IO.");
    file_ = H5::H5File(filepath, mode2flags(HDF5_FILE_MODE::READWRITE), {}, fapl);
    has_mpio_ = true;
    rank_ = comm.rank();
  }

  /// Write @p matrix into dataset @p dataset_name
  ///
  /// @pre if @p matrix is distributed, the file should support parallel-write
  template <class T, Device D>
  void write(matrix::Matrix<const T, D>& matrix, const std::string& dataset_name) const {
    matrix::MatrixMirror<const T, Device::CPU, D> matrix_mirror(matrix);

    matrix::Matrix<const T, Device::CPU>& matrix_host = matrix_mirror.get();

    const bool is_local_matrix = matrix::local_matrix(matrix_host);

    DLAF_ASSERT(is_local_matrix || has_mpio_,
                "You are trying to store a distributed matrix using a local only file", is_local_matrix,
                has_mpio_);

    const hsize_t dims_file[3] = {
        dlaf::to_sizet(matrix_host.size().cols()),
        dlaf::to_sizet(matrix_host.size().rows()),
        internal::hdf5_datatype<T>::dims,
    };
    H5::DataSpace dataspace_file(3, dims_file);

    // TODO it might be needed to wait all pika tasks
    H5::DataSet dataset =
        file_.createDataSet(dataset_name, internal::hdf5_datatype<T>::type, dataspace_file);

    if (!is_local_matrix || rank_ == 0)
      internal::to_dataset<T>(matrix_host, dataset);
  }

  /// Read dataset @p dataset_name in a local matrix with given @p blocksize.
  template <class T, Device D = Device::CPU>
  Matrix<T, D> read(const std::string& dataset_name, const TileElementSize blocksize) const {
    const H5::DataSet dataset = openDataSet<T>(dataset_name);

    const LocalElementSize size = FileHDF5::datasetToSize<LocalElementSize>(dataset);
    const matrix::Distribution dist(size, blocksize);

    matrix::Matrix<T, Device::CPU> mat(dist);
    internal::from_dataset<T>(dataset, mat);
    return returnMatrixOn<D>(std::move(mat));
  }

  /// Read dataset @p dataset_name in the matrix distributed accordingly to given parameters.
  template <class T, Device D = Device::CPU>
  Matrix<T, D> read(const std::string& dataset_name, const TileElementSize blocksize,
                    comm::CommunicatorGrid grid, const dlaf::comm::Index2D src_rank_index) const {
    const H5::DataSet dataset = openDataSet<T>(dataset_name);

    const GlobalElementSize size = FileHDF5::datasetToSize<GlobalElementSize>(dataset);
    const matrix::Distribution dist(size, blocksize, grid.size(), grid.rank(), src_rank_index);

    matrix::Matrix<T, Device::CPU> mat(dist);
    internal::from_dataset<T>(dataset, mat);
    return returnMatrixOn<D>(std::move(mat));
  }

private:
  static unsigned int mode2flags(const HDF5_FILE_MODE mode) {
    switch (mode) {
      case HDF5_FILE_MODE::READONLY:
        return H5F_ACC_RDONLY;
      case HDF5_FILE_MODE::READWRITE:
        return H5F_ACC_RDWR | H5F_ACC_CREAT;
    }
    return DLAF_UNREACHABLE(unsigned int);
  }

  template <class Index2D>
  static Index2D datasetToSize(const H5::DataSet& dataset) {
    const H5::DataSpace& dataspace = dataset.getSpace();
    DLAF_ASSERT(dataspace.getSimpleExtentNdims() == 3, dataspace.getSimpleExtentNdims());

    hsize_t dims_file[3];
    dataset.getSpace().getSimpleExtentDims(dims_file);

    return Index2D{to_SizeType(dims_file[1]), to_SizeType(dims_file[0])};
  }

  template <Device Target, class T, Device Source>
  static Matrix<T, Target> returnMatrixOn(Matrix<T, Source> source) {
    if constexpr (Source == Target)
      return source;
    else {
      Matrix<T, Target> target(source.distribution);
      copy(source, target);
      return target;
    }
  }

  template <class T>
  H5::DataSet openDataSet(const std::string& dataset_name) const {
    const H5::DataSet dataset = file_.openDataSet(dataset_name);

    const auto hdf5_t = internal::hdf5_datatype<BaseType<T>>::type;
    DLAF_ASSERT(hdf5_t == dataset.getDataType(), "HDF5 type mismatch");

    return dataset;
  }

  H5::H5File file_;
  bool has_mpio_ = false;
  comm::IndexT_MPI rank_ = 0;
};
}

#endif
