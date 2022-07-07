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
#include "dlaf/communication/communicator.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/memory/memory_view.h"

#include "dlaf_test/matrix/util_tile.h"

using namespace dlaf;
using namespace dlaf::matrix::test;

class P2PTest : public ::testing::Test {
  static_assert(NUM_MPI_RANKS >= 2, "at least 2 ranks are required");

protected:
  using T = int;
  static constexpr auto device = Device::CPU;

  comm::Communicator world = MPI_COMM_WORLD;
};

template <class T, Device device>
auto newBlockMatrixContiguous() {
  auto layout = matrix::colMajorLayout({13, 13}, {13, 13}, 13);
  auto dist = matrix::Distribution({13, 13}, {13, 13});

  auto matrix = matrix::Matrix<T, device>(dist, layout);

  EXPECT_TRUE(data_iscontiguous(common::make_data(matrix.read(LocalTileIndex(0, 0)).get())));

  return matrix;
}

template <class T, Device device>
auto newBlockMatrixStrided() {
  auto layout = matrix::colMajorLayout({13, 13}, {13, 13}, 26);
  auto dist = matrix::Distribution({13, 13}, {13, 13});

  auto matrix = matrix::Matrix<T, device>(dist, layout);

  EXPECT_FALSE(data_iscontiguous(common::make_data(matrix.read(LocalTileIndex(0, 0)).get())));

  return matrix;
}

template <class T, Device device>
void testSendRecv(comm::Communicator world, matrix::Matrix<T, device> matrix, std::string test_name) {
  constexpr comm::IndexT_MPI tag = 13;

  const LocalTileIndex idx(0, 0);

  const comm::IndexT_MPI rank_src = world.size() - 1;
  const comm::IndexT_MPI rank_dst = (world.size() - 1) / 2;

  auto input_tile = fixedValueTile(26);

  if (rank_src == world.rank()) {
    matrix::test::set(matrix(idx).get(), input_tile);
    comm::scheduleSend(rank_dst, world, tag, matrix.read_sender(idx));
  }
  else if (rank_dst == world.rank()) {
    matrix::test::set(matrix(idx).get(), fixedValueTile(13));
    comm::scheduleRecv(rank_src, world, tag, matrix.readwrite_sender(idx));
  }

  const auto& tile = matrix.read(idx).get();
  SCOPED_TRACE(test_name);
  CHECK_TILE_EQ(input_tile, tile);
}

TEST_F(P2PTest, SendRecv) {
  testSendRecv(world, newBlockMatrixContiguous<T, device>(), "Contiguous");
  testSendRecv(world, newBlockMatrixStrided<T, device>(), "Strided");
}
