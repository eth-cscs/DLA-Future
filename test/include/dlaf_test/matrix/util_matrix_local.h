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

/// @file

#include <functional>
#include <type_traits>

#include <dlaf/common/range2d.h>
#include <dlaf/communication/communicator_grid.h>
#include <dlaf/communication/functions_sync.h>
#include <dlaf/matrix/copy_tile.h>
#include <dlaf/matrix/index.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/print_numpy.h>
#include <dlaf/util_matrix.h>

#include <gtest/gtest.h>

#include <dlaf_test/matrix/matrix_local.h>

namespace dlaf {
namespace matrix {
namespace test {

/// Sets the elements of the matrix.
///
/// The (i, j)-element of the matrix is set to el({i, j}).
/// @pre el argument is an index of type const GlobalElementIndex& or GlobalElementIndex,
/// @pre el return type should be T.
template <class T, class ElementGetter>
void set(const MatrixLocal<T>& matrix, ElementGetter el) {
  using dlaf::common::iterate_range2d;

  for (const auto& tile_index : iterate_range2d(matrix.size()))
    matrix(tile_index) = el(tile_index);
}

template <class T>
void copy(const MatrixLocal<const T>& source, MatrixLocal<T>& dest) {
  DLAF_ASSERT(source.size() == dest.size(), source.size(), dest.size());

  const auto linear_size = static_cast<std::size_t>(source.size().rows() * source.size().cols());

  std::copy(source.ptr(), source.ptr() + linear_size, dest.ptr());
}

namespace internal {
auto checkerForIndexIn(const blas::Uplo uplo) {
  auto targeted_tile = [uplo](const GlobalTileIndex idx) {
    switch (uplo) {
      case blas::Uplo::General:
        return true;
      case blas::Uplo::Lower:
        return idx.row() >= idx.col();
      case blas::Uplo::Upper:
        return idx.row() <= idx.col();
      default:
        return false;
    }
  };

  return targeted_tile;
}
}

/// Given a local Matrix, it collects the full data locally, according @p to uplo
/// Optionally, it is possible to specify the type of the return MatrixLocal (useful for const correctness)
template <class T>
MatrixLocal<T> allGather(blas::Uplo uplo, Matrix<const T, Device::CPU>& source) {
  DLAF_ASSERT(matrix::local_matrix(source), source);

  MatrixLocal<std::remove_const_t<T>> dest(source.size(), source.baseTileSize());

  auto targeted_tile = internal::checkerForIndexIn(uplo);

  for (const auto& ij_tile : iterate_range2d(source.nrTiles())) {
    if (!targeted_tile(ij_tile))
      continue;

    auto& dest_tile = dest.tile(ij_tile);
    auto source_tile_holder = pika::this_thread::experimental::sync_wait(source.read(ij_tile));
    const auto& source_tile = source_tile_holder.get();
    matrix::internal::copy(source_tile, dest_tile);
  }

  return MatrixLocal<T>(std::move(dest));
}

/// Given a distributed Matrix, it collects the full data locally, according to @p uplo
/// Optionally, it is possible to specify the type of the return MatrixLocal (useful for const correctness)
template <class T>
MatrixLocal<T> allGather(blas::Uplo uplo, Matrix<const T, Device::CPU>& source,
                         comm::CommunicatorGrid& comm_grid) {
  DLAF_ASSERT(matrix::equal_process_grid(source, comm_grid), source, comm_grid);

  MatrixLocal<std::remove_const_t<T>> dest(source.size(), source.baseTileSize());

  const auto& dist_source = source.distribution();
  const auto rank = dist_source.rankIndex();

  auto targeted_tile = internal::checkerForIndexIn(uplo);

  for (const auto& ij_tile : iterate_range2d(dist_source.nrTiles())) {
    if (!targeted_tile(ij_tile))
      continue;

    const auto owner = dist_source.rankGlobalTile(ij_tile);

    auto& dest_tile = dest.tile(ij_tile);

    if (owner == rank) {
      const auto source_tile_holder = pika::this_thread::experimental::sync_wait(source.read(ij_tile));
      const auto& source_tile = source_tile_holder.get();
      comm::sync::broadcast::send(comm_grid.fullCommunicator(), source_tile);
      matrix::internal::copy(source_tile, dest_tile);
    }
    else {
      comm::sync::broadcast::receive_from(comm_grid.rankFullCommunicator(owner),
                                          comm_grid.fullCommunicator(), dest_tile);
    }
  }

  return MatrixLocal<T>(std::move(dest));
}

template <class T>
void print(format::numpy, std::string symbol, const MatrixLocal<const T>& matrix,
           std::ostream& os = std::cout) {
  using common::iterate_range2d;

  os << symbol << " = np.zeros(" << matrix.size() << ", dtype=np."
     << dlaf::matrix::internal::numpy_datatype<T>::typestring << ")\n";

  auto getTileTopLeft = [&](const GlobalTileIndex& ij) -> GlobalElementIndex {
    return {ij.row() * matrix.blockSize().rows(), ij.col() * matrix.blockSize().cols()};
  };

  for (const auto& ij : iterate_range2d(matrix.nrTiles())) {
    const auto& tile = matrix.tile_read(ij);

    const auto index_tl = getTileTopLeft(ij);

    os << symbol << "[" << index_tl.row() << ":" << index_tl.row() + tile.size().rows() << ","
       << index_tl.col() << ":" << index_tl.col() + tile.size().cols() << "] = ";

    print(format::numpy{}, tile, os);
  }
}
}
}
}
