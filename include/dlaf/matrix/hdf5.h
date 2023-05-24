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

template <class T>
void from_dataset(const H5::DataSet& dataset, dlaf::Matrix<T, Device::CPU>& matrix) {
  namespace tt = pika::this_thread::experimental;

  const H5::DataSpace& dataspace_file = dataset.getSpace();

  const auto& dist = matrix.distribution();
  for (const auto ij : common::iterate_range2d(dist.localNrTiles())) {
    auto tile = tt::sync_wait(matrix.readwrite(ij));

    const GlobalTileIndex ij_g = dist.globalTileIndex(ij);
    const GlobalElementIndex ij_e = dist.globalElementIndex(ij_g, {0, 0});

    // FILE DATASPACE
    {
      const hsize_t counts[3] = {
          to_sizet(tile.size().cols()),
          to_sizet(tile.size().rows()),
          internal::hdf5_datatype<T>::dims,
      };
      const hsize_t offsets[3] = {
          to_sizet(ij_e.col()),
          to_sizet(ij_e.row()),
          0,
      };
      dataspace_file.selectHyperslab(H5S_SELECT_SET, counts, offsets);
    }

    // MEMORY DATASPACE
    const hsize_t dims_mem[3] = {
        to_sizet(tile.size().cols()),
        to_sizet(tile.ld()),
        internal::hdf5_datatype<T>::dims,
    };
    H5::DataSpace dataspace_mem(3, dims_mem);

    {
      const hsize_t counts[3] = {
          to_sizet(tile.size().cols()),
          to_sizet(tile.size().rows()),
          internal::hdf5_datatype<T>::dims,
      };
      const hsize_t offsets[3] = {0, 0, 0};
      dataspace_mem.selectHyperslab(H5S_SELECT_SET, counts, offsets);
    }

    // read dataset
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

    const GlobalTileIndex ij_g = dist.globalTileIndex(ij);
    const GlobalElementIndex ij_e = dist.globalElementIndex(ij_g, {0, 0});

    // FILE DATASPACE
    {
      const hsize_t counts[3] = {
          to_sizet(tile.size().cols()),
          to_sizet(tile.size().rows()),
          internal::hdf5_datatype<T>::dims,
      };
      const hsize_t offsets[3] = {
          to_sizet(ij_e.col()),
          to_sizet(ij_e.row()),
          0,
      };
      dataspace_file.selectHyperslab(H5S_SELECT_SET, counts, offsets);
    }

    // MEMORY DATASPACE
    const hsize_t dims_mem[3] = {
        to_sizet(tile.size().cols()),
        to_sizet(tile.ld()),
        internal::hdf5_datatype<T>::dims,
    };
    H5::DataSpace dataspace_mem(3, dims_mem);

    {
      const hsize_t counts[3] = {
          to_sizet(tile.size().cols()),
          to_sizet(tile.size().rows()),
          internal::hdf5_datatype<T>::dims,
      };
      const hsize_t offsets[3] = {0, 0, 0};
      dataspace_mem.selectHyperslab(H5S_SELECT_SET, counts, offsets);
    }

    // write dataset
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
  // This call implies that no other rank are running the same code
  FileHDF5(const std::string& filepath, const HDF5_FILE_MODE& mode = HDF5_FILE_MODE::READONLY) {
    file_ = H5::H5File(filepath, mode2flags(mode));
  }

  FileHDF5(comm::Communicator comm, const std::string& filepath, const HDF5_FILE_MODE& mode) {
    H5::FileAccPropList fapl;
    DLAF_ASSERT(H5Pset_fapl_mpio(fapl.getId(), comm, MPI_INFO_NULL) >= 0, "Problem setting up MPI-IO.");
    file_ = H5::H5File(filepath, mode2flags(mode), {}, fapl);
    rank_ = comm.rank();
  }

  template <class T, Device D>
  void write(const std::string& name, matrix::Matrix<const T, D>& matrix) const {
    matrix::MatrixMirror<const T, Device::CPU, D> matrix_mirror(matrix);

    matrix::Matrix<const T, Device::CPU>& matrix_host = matrix_mirror.get();

    const bool is_local_matrix = matrix::local_matrix(matrix_host);

    const hsize_t dims_file[3] = {
        dlaf::to_sizet(matrix_host.size().cols()),
        dlaf::to_sizet(matrix_host.size().rows()),
        internal::hdf5_datatype<T>::dims,
    };
    H5::DataSpace dataspace_file(3, dims_file);

    // TODO it might be needed to wait all pika tasks
    H5::DataSet dataset = file_.createDataSet(name, internal::hdf5_datatype<T>::type, dataspace_file);

    if (!is_local_matrix || rank_ == 0)
      internal::to_dataset<T>(matrix_host, dataset);
  }

  template <class T, Device D = Device::CPU>
  auto read(const std::string& name, const TileElementSize blocksize) const {
    const H5::DataSet dataset = file_.openDataSet(name);

    DLAF_ASSERT(dataset.getDataType() == internal::hdf5_datatype<BaseType<T>>::type,
                "HDF5 Type mismatch");

    const LocalElementSize size = FileHDF5::datasetToSize<LocalElementSize>(dataset);
    const matrix::Distribution dist(size, blocksize);
    matrix::Matrix<T, D> mat(dist);

    {
      matrix::MatrixMirror<T, D, Device::CPU> matrix_mirror(mat);
      internal::from_dataset<T>(dataset, matrix_mirror.get());
    }

    return mat;
  }

  template <class T, Device D = Device::CPU>
  auto read(const std::string& name, const TileElementSize blocksize, comm::CommunicatorGrid grid,
            const dlaf::comm::Index2D src_rank_index = {0, 0}) const {
    const H5::DataSet dataset = file_.openDataSet(name);

    DLAF_ASSERT(dataset.getDataType() == internal::hdf5_datatype<BaseType<T>>::type,
                "HDF5 type mismatch");

    const GlobalElementSize size = FileHDF5::datasetToSize<GlobalElementSize>(dataset);
    const matrix::Distribution dist(size, blocksize, grid.size(), grid.rank(), src_rank_index);
    matrix::Matrix<T, D> mat(dist);

    {
      matrix::MatrixMirror<T, D, Device::CPU> matrix_mirror(mat);
      internal::from_dataset<T>(dataset, matrix_mirror.get());
    }

    return mat;
  }

private:
  static unsigned int mode2flags(const HDF5_FILE_MODE mode) {
    switch (mode) {
      case HDF5_FILE_MODE::READONLY:
        return H5F_ACC_RDONLY;
      case HDF5_FILE_MODE::READWRITE:
        return H5F_ACC_RDWR | H5F_ACC_EXCL;
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

  H5::H5File file_;
  comm::IndexT_MPI rank_ = 0;
};
}

#endif
