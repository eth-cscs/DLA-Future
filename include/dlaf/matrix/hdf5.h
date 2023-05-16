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
#include "dlaf/types.h"

namespace dlaf::matrix {

namespace internal {

template <class T>
struct hdf5_datatype;

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

// clang-format off
template <>         const H5::PredType& hdf5_datatype<float>::type            = H5::PredType::NATIVE_FLOAT;
template <>         const H5::PredType& hdf5_datatype<double>::type           = H5::PredType::NATIVE_DOUBLE;
template <class T>  const H5::PredType& hdf5_datatype<std::complex<T>>::type  = hdf5_datatype<T>::type;
// clang-format on

}

H5::H5File open_hdf5file(dlaf::comm::Communicator comm, const std::string& filepath,
                         unsigned int flags = H5F_ACC_RDONLY) {
  H5::FileAccPropList fapl;
  DLAF_ASSERT(H5Pset_fapl_mpio(fapl.getId(), comm, MPI_INFO_NULL) >= 0, "Problem setting up MPI-IO.");
  return H5::H5File(filepath, flags, {}, fapl);
}

template <class T>
dlaf::matrix::Matrix<T, dlaf::Device::CPU> from_hdf5(const H5::H5File& file, const std::string& name,
                                                     const TileElementSize blocksize,
                                                     comm::CommunicatorGrid grid,
                                                     const dlaf::comm::Index2D src_rank_index = {0, 0}) {
  namespace tt = pika::this_thread::experimental;

  auto dataset = file.openDataSet(name);

  auto dataspace_file = dataset.getSpace();
  DLAF_ASSERT(dataspace_file.getSimpleExtentNdims() == 3, dataspace_file.getSimpleExtentNdims());

  hsize_t dims_file[3];
  dataspace_file.getSimpleExtentDims(dims_file);
  DLAF_ASSERT(dims_file[2] == internal::hdf5_datatype<T>::dims, dims_file[2],
              internal::hdf5_datatype<T>::dims);

  const GlobalElementSize size(to_SizeType(dims_file[1]), to_SizeType(dims_file[0]));
  const matrix::Distribution dist(size, blocksize, grid.size(), grid.rank(), src_rank_index);
  matrix::Matrix<T, dlaf::Device::CPU> mat(dist);

  for (const auto ij : common::iterate_range2d(dist.localNrTiles())) {
    auto tile = tt::sync_wait(mat.readwrite(ij));

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
    dataset.read(tile.ptr(), internal::hdf5_datatype<T>::type, dataspace_mem, dataspace_file);
  }

  return mat;
}

template <class T>
auto to_hdf5(const H5::H5File& file, const std::string& name,
             dlaf::matrix::Matrix<T, dlaf::Device::CPU>& mat) {
  namespace tt = pika::this_thread::experimental;

  const hsize_t dims_file[3] = {
      dlaf::to_sizet(mat.size().cols()),
      dlaf::to_sizet(mat.size().rows()),
      internal::hdf5_datatype<T>::dims,
  };
  H5::DataSpace dataspace_file(3, dims_file);

  auto dataset = file.createDataSet(name, internal::hdf5_datatype<T>::type, dataspace_file);

  const auto& dist = mat.distribution();
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

#endif
