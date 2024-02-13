//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <algorithm>

#include <pika/execution.hpp>

#include <dlaf/common/index2d.h>
#include <dlaf/common/range2d.h>
#include <dlaf/communication/communicator_grid.h>
#include <dlaf/communication/sync/broadcast.h>
#include <dlaf/lapack/tile.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/tile.h>
#include <dlaf/memory/memory_view.h>
#include <dlaf/types.h>

#include <gtest/gtest.h>

#include <dlaf_test/matrix/matrix_local.h>
#include <dlaf_test/matrix/util_matrix.h>

namespace dlaf::test {

template <class T>
auto allGatherTaus(const SizeType k, Matrix<T, Device::CPU>& mat_local_taus) {
  namespace tt = pika::this_thread::experimental;
  namespace ex = pika::execution::experimental;

  auto local_taus_tiles = tt::sync_wait(ex::when_all_vector(selectRead(
      mat_local_taus, common::iterate_range2d(LocalTileSize(mat_local_taus.nrTiles().rows(), 1)))));

  std::vector<T> taus;
  taus.reserve(to_sizet(k));
  for (const auto& t : local_taus_tiles) {
    std::copy(t.get().ptr(), t.get().ptr() + t.get().size().rows(), std::back_inserter(taus));
  }

  DLAF_ASSERT(to_SizeType(taus.size()) == k, taus.size(), k);

  return taus;
}

template <class T>
auto allGatherTaus(const SizeType k, Matrix<T, Device::CPU>& mat_taus,
                   comm::CommunicatorGrid& comm_grid) {
  namespace tt = pika::this_thread::experimental;

  const auto local_num_tiles = mat_taus.distribution().localNrTiles().rows();
  const auto num_tiles = mat_taus.distribution().nrTiles().rows();
  const auto local_num_tiles_expected =
      num_tiles / comm_grid.size().cols() +
      (comm_grid.rank().col() < (num_tiles % comm_grid.size().cols()) ? 1 : 0);
  EXPECT_EQ(local_num_tiles, local_num_tiles_expected);

  std::vector<T> taus;
  taus.reserve(to_sizet(k));

  for (SizeType i = 0; i < mat_taus.nrTiles().rows(); ++i) {
    const auto owner = mat_taus.rankGlobalTile(GlobalTileIndex(i, 0)).row();
    const bool is_owner = owner == comm_grid.rank().col();

    const auto chunk_size = mat_taus.tileSize(GlobalTileIndex(i, 0)).rows();

    if (is_owner) {
      auto tile_local = tt::sync_wait(mat_taus.read(GlobalTileIndex(i, 0)));
      dlaf::comm::sync::broadcast::send(comm_grid.rowCommunicator(),
                                        common::make_data(tile_local.get()));
      std::copy(tile_local.get().ptr(), tile_local.get().ptr() + tile_local.get().size().rows(),
                std::back_inserter(taus));
    }
    else {
      dlaf::matrix::Tile<T, Device::CPU> tile_local(TileElementSize(chunk_size, 1),
                                                    dlaf::memory::MemoryView<T, Device::CPU>(chunk_size),
                                                    chunk_size);
      dlaf::comm::sync::broadcast::receive_from(owner, comm_grid.rowCommunicator(),
                                                common::make_data(tile_local));
      std::copy(tile_local.ptr(), tile_local.ptr() + tile_local.size().rows(), std::back_inserter(taus));
    }
  }

  return taus;
}

template <class T>
auto checkResult(const SizeType k, const SizeType band_size, Matrix<const T, Device::CPU>& reference,
                 const dlaf::matrix::test::MatrixLocal<T>& mat_v,
                 const dlaf::matrix::test::MatrixLocal<T>& mat_b, const std::vector<T>& taus) {
  const GlobalElementIndex offset(band_size, 0);
  // Now that all input are collected locally, it's time to apply the transformation,
  // ...but just if there is any
  if (offset.isIn(mat_v.size())) {
    // Reduction to band returns a sequence of transformations applied from left and right to A
    // allowing to reduce the matrix A to a band matrix B
    //
    // Hn* ... H2* H1* A H1 H2 ... Hn
    // Q* A Q = B
    //
    // Applying the inverse of the same transformations, we can go from B to A
    // Q B Q* = A
    // Q = H1 H2 ... Hn
    // H1 H2 ... Hn B Hn* ... H2* H1*

    dlaf::common::internal::SingleThreadedBlasScope single;

    // apply from left...
    const GlobalElementIndex left_offset = offset;
    const GlobalElementSize left_size{mat_b.size().rows() - band_size, mat_b.size().cols()};
    lapack::unmqr(lapack::Side::Left, lapack::Op::NoTrans, left_size.rows(), left_size.cols(), k,
                  mat_v.ptr(offset), mat_v.ld(), taus.data(), mat_b.ptr(left_offset), mat_b.ld());

    // ... and from right
    const GlobalElementIndex right_offset = common::transposed(left_offset);
    const GlobalElementSize right_size = common::transposed(left_size);

    lapack::unmqr(lapack::Side::Right, lapack::Op::ConjTrans, right_size.rows(), right_size.cols(), k,
                  mat_v.ptr(offset), mat_v.ld(), taus.data(), mat_b.ptr(right_offset), mat_b.ld());
  }

  // Eventually, check the result obtained by applying the inverse transformation equals the original matrix
  auto result = [&dist = reference.distribution(),
                 &mat_local = mat_b](const GlobalElementIndex& element) {
    const auto tile_index = dist.globalTileIndex(element);
    const auto tile_element = dist.tileElementIndex(element);
    return mat_local.tile_read(tile_index)(tile_element);
  };

  CHECK_MATRIX_NEAR(result, reference, 0,
                    std::max<SizeType>(1, mat_b.size().linear_size()) * TypeUtilities<T>::error);
}

}
