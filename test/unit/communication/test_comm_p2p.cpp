//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/communication/kernels/p2p.h"
#include "dlaf/communication/kernels/p2p_allsum.h"

#include <gtest/gtest.h>

#include "dlaf/common/data.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/range2d.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/matrix/copy_tile.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/index.h"
#include "dlaf/matrix/layout_info.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/memory/memory_view.h"
#include "dlaf/sender/policy.h"
#include "dlaf/sender/transform.h"
#include "dlaf/sender/when_all_lift.h"
#include "dlaf/types.h"

#include "dlaf_test/matrix/util_tile.h"

using namespace dlaf;
using dlaf::matrix::test::fixedValueTile;
using dlaf::matrix::test::set;

namespace ex = pika::execution::experimental;
namespace tt = pika::this_thread::experimental;

template <Device D>
class P2PTest : public ::testing::Test {
  static_assert(NUM_MPI_RANKS >= 2, "at least 2 ranks are required");

protected:
  using T = float;
  using MatrixT = matrix::Matrix<T, D>;

  comm::Communicator world = MPI_COMM_WORLD;
};

using P2PTestMC = P2PTest<Device::CPU>;

#ifdef DLAF_WITH_GPU
using P2PTestGPU = P2PTest<Device::GPU>;
#endif

template <class T, class SenderTile>
auto setTileTo(SenderTile&& tile, const T input_value) {
  constexpr auto D = internal::SenderSingleValueType<SenderTile>::device;
  return internal::whenAllLift(blas::Uplo::General, input_value, input_value,
                               std::forward<SenderTile>(tile)) |
         tile::laset(internal::Policy<DefaultBackend_v<D>>());
}

template <class SenderTile, class TileLike>
auto checkTileEq(TileLike&& ref_tile, SenderTile&& tile) {
  constexpr auto D = internal::SenderSingleValueType<SenderTile>::device;
  return std::forward<SenderTile>(tile) |
         internal::transform(internal::Policy<DefaultBackend_v<D>>(), matrix::Duplicate<Device::CPU>{}) |
         ex::then([&](const auto& tile_cpu) { CHECK_TILE_EQ(ref_tile, tile_cpu); });
}

template <class T, Device D>
void testSendRecv(comm::Communicator world, matrix::Matrix<T, D> matrix) {
  const LocalTileIndex idx(0, 0);

  const comm::IndexT_MPI rank_src = world.size() - 1;
  const comm::IndexT_MPI rank_dst = (world.size() - 1) / 2;

  constexpr comm::IndexT_MPI tag = 13;

  const T input_value = 26;
  const auto input_tile = fixedValueTile<T>(input_value);

  if (rank_src == world.rank()) {
    tt::sync_wait(setTileTo(matrix.readwrite_sender(idx), input_value));
    ex::start_detached(comm::scheduleSend(ex::make_unique_any_sender(ex::just(world)), rank_dst, tag,
                                          ex::make_unique_any_sender(matrix.read_sender(idx))));
  }
  else if (rank_dst == world.rank()) {
    ex::start_detached(comm::scheduleRecv(ex::make_unique_any_sender(ex::just(world)), rank_src, tag,
                                          ex::make_unique_any_sender(matrix.readwrite_sender(idx))));
  }
  else {
    return;
  }

  tt::sync_wait(checkTileEq(input_tile, matrix.read_sender(idx)));
}

TEST_F(P2PTestMC, SendRecv) {
  auto dist = matrix::Distribution({13, 13}, {13, 13});

  // single tile matrix whose columns are stored in contiguous memory
  testSendRecv(world, MatrixT(dist, matrix::tileLayout(dist, 13, 13)));

  // single tile matrix whose columns are stored in non-contiguous memory
  testSendRecv(world, MatrixT(dist, matrix::colMajorLayout(dist, 13)));
}

#ifdef DLAF_WITH_GPU
TEST_F(P2PTestGPU, SendRecv) {
  auto dist = matrix::Distribution({13, 13}, {13, 13});

  // single tile matrix whose columns are stored in contiguous memory
  testSendRecv(world, MatrixT(dist, matrix::tileLayout(dist, 13, 13)));

  // single tile matrix whose columns are stored in non-contiguous memory
  testSendRecv(world, MatrixT(dist, matrix::colMajorLayout(dist, 13)));
}
#endif

template <class T, Device D>
void testSendRecvMixTags(comm::Communicator world, matrix::Matrix<T, D> matrix) {
  // This test involves just 2 ranks, where rank_src sends all tiles allowing to "mirror" the
  // entire matrix on rank_dst. P2P communications are issued by the different ranks in different
  // orders, linking them using the tag of the MPI communication.
  const comm::IndexT_MPI rank_src = world.size() - 1;
  const comm::IndexT_MPI rank_dst = (world.size() - 1) / 2;

  if (rank_src == world.rank()) {
    // rank_src sends tile by tile starting from the top left and following a column major order
    for (SizeType c = 0; c < matrix.nrTiles().cols(); ++c) {
      for (SizeType r = 0; r < matrix.nrTiles().rows(); ++r) {
        const GlobalTileIndex idx(r, c);
        const auto id = common::computeLinearIndexColMajor<comm::IndexT_MPI>(idx, matrix.nrTiles());
        matrix::test::set(matrix(idx).get(), fixedValueTile(id));
        ex::start_detached(comm::scheduleSend(ex::make_unique_any_sender(ex::just(world)), rank_dst, id,
                                              ex::make_unique_any_sender(matrix.read_sender(idx))));
      }
    }
  }
  else if (rank_dst == world.rank()) {
    // rank_ds receives all tiles, issuing recv requests in a different order w.r.t how they
    // are transimetted by rank_src. Indeed, requests are issued following a sort-of "inverse"
    // column-major order, where the first transmitted tile is the bottom right one, then the
    // one above it in the same column follows, and so on.
    for (SizeType r = matrix.nrTiles().rows() - 1; r >= 0; --r) {
      for (SizeType c = matrix.nrTiles().cols() - 1; c >= 0; --c) {
        const GlobalTileIndex idx(r, c);
        const auto id = common::computeLinearIndexColMajor<comm::IndexT_MPI>(idx, matrix.nrTiles());
        ex::start_detached(comm::scheduleRecv(ex::make_unique_any_sender(ex::just(world)), rank_src, id,
                                              ex::make_unique_any_sender(matrix.readwrite_sender(idx))));
      }
    }
  }
  else {
    // other ranks are not involved
    return;
  }

  // In the end the full matrix should be the same way on both ranks. This test does not want to
  // test MPI communications tag functionality (e.g. by checking that after a tagged communication
  // finishes, exactly that tile has been populated on the receiving endpoint), but it wants to
  // check that overall this mechanism work also with unordered tagged communications.
  for (const GlobalTileIndex idx : common::iterate_range2d(matrix.nrTiles())) {
    const auto id = common::computeLinearIndexColMajor<comm::IndexT_MPI>(idx, matrix.nrTiles());
    CHECK_TILE_EQ(fixedValueTile(id), matrix.read(idx).get());
  }
}

TEST_F(P2PTestMC, SendRecvMixTags) {
  const auto dist = matrix::Distribution({10, 10}, {3, 3});

  // each tile is stored in contiguous memory (i.e. ld == blocksize.rows())
  testSendRecvMixTags(world, MatrixT(dist, matrix::tileLayout(dist, 3, 4)));

  // tiles are stored in non-contiguous memory
  testSendRecvMixTags(world, MatrixT(dist, matrix::colMajorLayout(dist, 10)));
}

template <Backend B, Device D, class T>
void testP2PAllSum(comm::Communicator world, matrix::Matrix<T, D> matrix) {
  const LocalTileIndex idx(0, 0);

  const comm::IndexT_MPI rank_src = world.size() - 1;
  const comm::IndexT_MPI rank_dst = (world.size() - 1) / 2;

  constexpr comm::IndexT_MPI tag = 13;

  const T input_value = 13;
  tt::sync_wait(setTileTo(matrix.readwrite_sender(idx), input_value));

  matrix::Matrix<T, D> tmp(matrix.distribution().localSize(), matrix.blockSize());

  if (rank_src == world.rank()) {
    ex::start_detached(comm::scheduleAllSumP2P<B>(ex::just(world), rank_dst, tag,
                                                  matrix.read_sender(idx),
                                                  tmp.readwrite_sender(LocalTileIndex{0, 0})));
  }
  else if (rank_dst == world.rank()) {
    ex::start_detached(comm::scheduleAllSumP2P<B>(ex::just(world), rank_src, tag,
                                                  matrix.read_sender(idx),
                                                  tmp.readwrite_sender(LocalTileIndex{0, 0})));
  }
  else {
    return;
  }

  tt::sync_wait(checkTileEq(fixedValueTile(26), tmp.read_sender(idx)));
}

TEST_F(P2PTestMC, AllSum) {
  auto dist = matrix::Distribution({13, 13}, {13, 13});

  // single tile matrix whose columns are stored in contiguous memory
  testP2PAllSum<Backend::MC>(world, MatrixT(dist, matrix::tileLayout(dist, 13, 13)));

  // single tile matrix whose columns are stored in non-contiguous memory
  testP2PAllSum<Backend::MC>(world, MatrixT(dist, matrix::colMajorLayout(dist, 13)));
}

#ifdef DLAF_WITH_GPU
TEST_F(P2PTestGPU, AllSum) {
  auto dist = matrix::Distribution({13, 13}, {13, 13});

  // single tile matrix whose columns are stored in contiguous memory
  testP2PAllSum<Backend::GPU>(world, MatrixT(dist, matrix::tileLayout(dist, 13, 13)));

  // single tile matrix whose columns are stored in non-contiguous memory
  testP2PAllSum<Backend::GPU>(world, MatrixT(dist, matrix::colMajorLayout(dist, 13)));
}
#endif

template <Backend B, Device D, class T>
void testP2PAllSumMixTags(comm::Communicator world, matrix::Matrix<T, D> matrix) {
  // This test involves just 2 ranks, where rank_src sends all tiles allowing to "mirror" the
  // entire matrix on rank_dst. P2P communications are issued by the different ranks in different
  // orders, linking them using the tag of the MPI communication.
  const comm::IndexT_MPI rank_src = world.size() - 1;
  const comm::IndexT_MPI rank_dst = (world.size() - 1) / 2;

  matrix::Matrix<T, D> tmp(matrix.distribution().localSize(), matrix.blockSize());

  if (rank_src == world.rank()) {
    // rank_src sends tile by tile starting from the top left and following a column major order
    for (SizeType c = 0; c < matrix.nrTiles().cols(); ++c) {
      for (SizeType r = 0; r < matrix.nrTiles().rows(); ++r) {
        const GlobalTileIndex idx(r, c);
        const auto id = common::computeLinearIndexColMajor<comm::IndexT_MPI>(idx, matrix.nrTiles());
        matrix::test::set(matrix(idx).get(), fixedValueTile(id));
        ex::start_detached(comm::scheduleAllSumP2P<B>(ex::just(world), rank_dst, id,
                                                      matrix.read_sender(idx),
                                                      tmp.readwrite_sender(idx)));
      }
    }
  }
  else if (rank_dst == world.rank()) {
    // rank_ds receives all tiles, issuing recv requests in a different order w.r.t how they
    // are transimetted by rank_src. Indeed, requests are issued following a sort-of "inverse"
    // column-major order, where the first transmitted tile is the bottom right one, then the
    // one above it in the same column follows, and so on.
    for (SizeType r = matrix.nrTiles().rows() - 1; r >= 0; --r) {
      for (SizeType c = matrix.nrTiles().cols() - 1; c >= 0; --c) {
        const GlobalTileIndex idx(r, c);
        const auto id = common::computeLinearIndexColMajor<comm::IndexT_MPI>(idx, matrix.nrTiles());
        matrix::test::set(matrix(idx).get(), fixedValueTile(id));
        ex::start_detached(comm::scheduleAllSumP2P<B>(ex::just(world), rank_src, id,
                                                      matrix.read_sender(idx),
                                                      tmp.readwrite_sender(idx)));
      }
    }
  }
  else {
    // other ranks are not involved
    return;
  }

  // In the end the full matrix should be the same way on both ranks. This test does not want to
  // test MPI communications tag functionality (e.g. by checking that after a tagged communication
  // finishes, exactly that tile has been populated on the receiving endpoint), but it wants to
  // check that overall this mechanism work also with unordered tagged communications.
  for (const GlobalTileIndex idx : common::iterate_range2d(matrix.nrTiles())) {
    const auto id = common::computeLinearIndexColMajor<comm::IndexT_MPI>(idx, matrix.nrTiles());
    CHECK_TILE_EQ(fixedValueTile(2 * id), tmp.read(idx).get());
  }
}

TEST_F(P2PTestMC, AllSumMixTags) {
  const auto dist = matrix::Distribution({10, 10}, {3, 3});

  // each tile is stored in contiguous memory (i.e. ld == blocksize.rows())
  testP2PAllSumMixTags<Backend::MC>(world, MatrixT(dist, matrix::tileLayout(dist, 3, 4)));

  // tiles are stored in non-contiguous memory
  testP2PAllSumMixTags<Backend::MC>(world, MatrixT(dist, matrix::colMajorLayout(dist, 10)));
}
