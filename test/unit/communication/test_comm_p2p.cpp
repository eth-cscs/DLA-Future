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

#include <gtest/gtest.h>
#include <mpi.h>

#include "dlaf/common/data.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/range2d.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/index.h"
#include "dlaf/matrix/layout_info.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/memory/memory_view.h"
#include "dlaf/types.h"

#include "dlaf_test/matrix/util_tile.h"

using namespace dlaf;
using namespace dlaf::matrix::test;

class P2PTest : public ::testing::Test {
  static_assert(NUM_MPI_RANKS >= 2, "at least 2 ranks are required");

protected:
  using T = int;
  using MatrixT = matrix::Matrix<T, Device::CPU>;

  comm::Communicator world = MPI_COMM_WORLD;
};

template <class T, Device device>
void testSendRecv(comm::Communicator world, matrix::Matrix<T, device> matrix) {
  namespace ex = pika::execution::experimental;

  const LocalTileIndex idx(0, 0);

  const comm::IndexT_MPI rank_src = world.size() - 1;
  const comm::IndexT_MPI rank_dst = (world.size() - 1) / 2;

  constexpr comm::IndexT_MPI tag = 13;

  auto input_tile = fixedValueTile(26);

  if (rank_src == world.rank()) {
    matrix::test::set(matrix(idx).get(), input_tile);
    ex::start_detached(comm::scheduleSend(rank_dst, world, tag, matrix.read_sender(idx)));
  }
  else if (rank_dst == world.rank()) {
    ex::start_detached(comm::scheduleRecv(rank_src, world, tag, matrix.readwrite_sender(idx)));
  }
  else {
    return;
  }

  CHECK_TILE_EQ(input_tile, matrix.read(idx).get());
}

TEST_F(P2PTest, SendRecv) {
  auto dist = matrix::Distribution({13, 13}, {13, 13});

  // single tile matrix whose columns are stored in contiguous memory
  testSendRecv(world, MatrixT(dist, matrix::tileLayout(dist, 13, 13)));

  // single tile matrix whose columns are stored in non-contiguous memory
  testSendRecv(world, MatrixT(dist, matrix::colMajorLayout(dist, 13)));
}

template <class T, Device device>
void testSendRecvMixTags(comm::Communicator world, matrix::Matrix<T, device> matrix) {
  namespace ex = pika::execution::experimental;

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
        ex::start_detached(comm::scheduleSend(rank_dst, world, id, matrix.read_sender(idx)));
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
        ex::start_detached(comm::scheduleRecv(rank_src, world, id, matrix.readwrite_sender(idx)));
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

TEST_F(P2PTest, SendRecvMixTags) {
  const auto dist = matrix::Distribution({10, 10}, {3, 3});

  // each tile is stored in contiguous memory (i.e. ld == blocksize.rows())
  testSendRecvMixTags(world, MatrixT(dist, matrix::tileLayout(dist, 3, 4)));

  // tiles are stored in non-contiguous memory
  testSendRecvMixTags(world, MatrixT(dist, matrix::colMajorLayout(dist, 10)));
}
