//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#ifdef DLAF_WITH_HDF5

#include <complex>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <typeinfo>
#include <utility>

#include <H5Cpp.h>
#include <mpi.h>

#include "dlaf/common/assert.h"
#include "dlaf/common/range2d.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/index.h"
#include "dlaf/matrix/index.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/matrix_mirror.h"
#include "dlaf/sender/policy.h"
#include "dlaf/sender/transform.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

namespace dlaf::matrix::internal {

// Type mappings
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

// Type to string mappings
template <typename T>
struct TypeToString {
  static inline constexpr std::string_view value = typeid(T).name();
};

template <typename T>
inline constexpr std::string_view TypeToString_v = TypeToString<T>::value;

template <>
struct TypeToString<float> {
  static inline constexpr std::string_view value = "s";
};

template <>
struct TypeToString<double> {
  static inline constexpr std::string_view value = "d";
};

template <>
struct TypeToString<std::complex<float>> {
  static inline constexpr std::string_view value = "c";
};

template <>
struct TypeToString<std::complex<double>> {
  static inline constexpr std::string_view value = "z";
};

// Helper function that for each local tile index in @p dist, gets a sender of a tile with
// @p get_tile and sends it to a function that takes care of the mapping between file and memory.
// Then, this function, passes all required arguments to @p dataset_op which should be either
// dataset.read or dataset.write, so it has the following signature
//    dataset_op(memory_ptr, type, dataspace_file, dataspace_mem)
template <class T, class TileGetter, class DatasetOp>
void for_each_local_map_with_dataset(Distribution dist, const H5::DataSet& dataset,
                                     TileGetter&& get_tile, DatasetOp&& dataset_op) {
  namespace tt = pika::this_thread::experimental;
  namespace di = dlaf::internal;

  const H5::DataSpace& ds_file = dataset.getSpace();

  for (const auto ij : common::iterate_range2d(dist.localNrTiles())) {
    auto map_ds_tile = [&](auto&& tile) {
      const GlobalElementIndex tl = dist.globalElementIndex(dist.globalTileIndex(ij), {0, 0});

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
      ds_file.selectHyperslab(H5S_SELECT_SET, file_counts, file_offsets);

      const hsize_t memory_dims[3] = {
          to_sizet(tile.size().cols()),
          to_sizet(tile.ld()),
          internal::hdf5_datatype<T>::dims,
      };
      H5::DataSpace ds_memory(3, memory_dims);

      const hsize_t memory_counts[3] = {
          to_sizet(tile.size().cols()),
          to_sizet(tile.size().rows()),
          internal::hdf5_datatype<T>::dims,
      };
      const hsize_t memory_offsets[3] = {0, 0, 0};
      ds_memory.selectHyperslab(H5S_SELECT_SET, memory_counts, memory_offsets);

      dataset_op(tile.ptr(), internal::hdf5_datatype<T>::type, ds_memory, ds_file);
    };

    tt::sync_wait(get_tile(ij) | di::transform(di::Policy<Backend::MC>(), std::move(map_ds_tile)));
  }
}

template <class T>
void from_dataset(const H5::DataSet& dataset, dlaf::Matrix<T, Device::CPU>& matrix) {
  auto tile_rw = [&matrix](LocalTileIndex ij) { return matrix.readwrite(ij); };
  auto dataset_read = [&dataset](auto&&... args) {
    dataset.read(std::forward<decltype(args)>(args)...);
  };
  for_each_local_map_with_dataset<T>(matrix.distribution(), dataset, std::move(tile_rw),
                                     std::move(dataset_read));
}

template <class T>
void to_dataset(dlaf::Matrix<const T, Device::CPU>& matrix, const H5::DataSet& dataset) {
  auto tile_ro = [&matrix](LocalTileIndex ij) { return matrix.read(ij); };
  auto dataset_write = [&dataset](auto&&... args) {
    dataset.write(std::forward<decltype(args)>(args)...);
  };
  for_each_local_map_with_dataset<T>(matrix.distribution(), dataset, std::move(tile_ro),
                                     std::move(dataset_write));
}

class FileHDF5 final {
public:
  /// File access modes:
  /// - readonly, the file will be opened (if should already exist)
  /// - readwrite, the file will be created (if should not exist)
  enum class FileMode {
    readonly,
    readwrite,
  };

  /// Create/open a local file.
  ///
  /// @p filepath filepath where the file to be opened is or where it will be created
  /// @p mode file access mode (see FileMode for more details)
  ///
  /// @post if mode == READWRITE, being a local file it will not support parallel-write
  FileHDF5(const std::string& filepath, const FileMode& mode) {
    file_ = H5::H5File(filepath, mode2flags(mode));
  }

  /// Create a file that, besides read, supports writing in parallel from different ranks.
  ///
  /// @p comm Communicator grouping all ranks that will be able to write in parallel to the file
  /// @p filepath filepath where the file will be created
  ///
  /// @pre @p filepath should not exist
  /// @post file created will support parallel-write
  FileHDF5(comm::Communicator comm, const std::string& filepath) {
    H5::FileAccPropList fapl;
    DLAF_ASSERT(H5Pset_fapl_mpio(fapl.getId(), comm, MPI_INFO_NULL) >= 0, "Problem setting up MPI-IO.");
    file_ = H5::H5File(filepath, mode2flags(FileMode::readwrite), {}, fapl);
    has_mpio_ = true;
    rank_ = comm.rank();
  }

  /// Write @p matrix into dataset @p dataset_name
  ///
  /// If the matrix is local, just one rank is going to write on the file.
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
                    comm::CommunicatorGrid& grid, const dlaf::comm::Index2D src_rank_index) const {
    const H5::DataSet dataset = openDataSet<T>(dataset_name);

    const GlobalElementSize size = FileHDF5::datasetToSize<GlobalElementSize>(dataset);
    const matrix::Distribution dist(size, blocksize, grid.size(), grid.rank(), src_rank_index);

    matrix::Matrix<T, Device::CPU> mat(dist);
    internal::from_dataset<T>(dataset, mat);
    return returnMatrixOn<D>(std::move(mat));
  }

  void flush() const {
    file_.flush(H5F_SCOPE_LOCAL);
  }

private:
  static unsigned int mode2flags(const FileMode mode) {
    switch (mode) {
      case FileMode::readonly:
        return H5F_ACC_RDONLY;
      case FileMode::readwrite:
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

  template <class T>
  H5::DataSet openDataSet(const std::string& dataset_name) const {
    const H5::DataSet dataset = file_.openDataSet(dataset_name);

    const auto hdf5_t = internal::hdf5_datatype<BaseType<T>>::type;
    DLAF_ASSERT(hdf5_t == dataset.getDataType(), "HDF5 type mismatch");

    return dataset;
  }

  // This helper returns @p source on specified @p Target device.
  // If @p source is not on @p Target, it clones it on @p Target and returns it. Otherwise, if it
  // was already on @p Target, it is simply returned.
  template <Device Target, class T, Device Source>
  static Matrix<T, Target> returnMatrixOn(Matrix<T, Source> source) {
    if constexpr (Source == Target)
      return source;

    Matrix<T, Target> target(source.distribution());
    copy(source, target);
    return target;
  }

  H5::H5File file_;
  bool has_mpio_ = false;
  comm::IndexT_MPI rank_ = 0;
};
}

#endif
