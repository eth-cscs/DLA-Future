//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/communication/kernels/all_reduce.h"
#include "dlaf/communication/kernels/reduce.h"

#include <gtest/gtest.h>
#include <mpi.h>

#include "dlaf/common/data.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/executor.h"
#include "dlaf/executors.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/memory/memory_view.h"

#include "dlaf_test/matrix/util_tile.h"

using namespace dlaf;

class CollectiveTest : public ::testing::Test {
  static_assert(NUM_MPI_RANKS >= 2, "at least 2 ranks are required");

protected:
  using T = int;
  static constexpr auto device = Device::CPU;

  comm::Communicator world = MPI_COMM_WORLD;
};

template <class T>
auto fixedValueTile(const T value) {
  return [value](TileElementIndex const&) { return value; };
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
void testReduceInPlace(comm::Communicator world, matrix::Matrix<T, device> matrix) {
  common::Pipeline<comm::Communicator> chain(world);

  const auto root_rank = world.size() - 1;
  const auto ex_mpi = dlaf::getMPIExecutor<Backend::MC>();

  const LocalTileIndex idx(0, 0);

  auto input_tile = fixedValueTile(world.rank() + 1);
  matrix::test::set(matrix(idx).get(), input_tile);

  if (root_rank == world.rank()) {
    // use -> read
    scheduleReduceRecvInPlace(ex_mpi, chain(), MPI_SUM, matrix(idx));

    auto exp_tile = fixedValueTile(world.size() * (world.size() + 1) / 2);
    CHECK_TILE_EQ(exp_tile, matrix.read(idx).get());
  }
  else {
    // use -> read -> set -> read
    scheduleReduceSend(ex_mpi, root_rank, chain(), MPI_SUM, matrix.read(idx));

    CHECK_TILE_EQ(input_tile, matrix.read(idx).get());

    auto new_tile = fixedValueTile(26);
    matrix::test::set(matrix(idx).get(), new_tile);
    CHECK_TILE_EQ(new_tile, matrix.read(idx).get());
  }
}

TEST_F(CollectiveTest, ReduceInPlace) {
  testReduceInPlace(world, newBlockMatrixContiguous<T, device>());
  testReduceInPlace(world, newBlockMatrixStrided<T, device>());
}

template <class T, Device device>
void testAllReduceInPlace(comm::Communicator world, matrix::Matrix<T, device> matrix) {
  common::Pipeline<comm::Communicator> chain(world);

  const auto ex_mpi = dlaf::getMPIExecutor<Backend::MC>();

  const LocalTileIndex idx(0, 0);

  // set -> use -> read
  auto input_tile = fixedValueTile(world.rank() + 1);
  matrix::test::set(matrix(idx).get(), input_tile);

  auto after = scheduleAllReduceInPlace(ex_mpi, chain(), MPI_SUM, matrix(idx));

  auto exp_tile = fixedValueTile(world.size() * (world.size() + 1) / 2);
  CHECK_TILE_EQ(exp_tile, after.get());
  CHECK_TILE_EQ(exp_tile, matrix.read(idx).get());
}

TEST_F(CollectiveTest, AllReduceInPlace) {
  testAllReduceInPlace(world, newBlockMatrixContiguous<T, device>());
  testAllReduceInPlace(world, newBlockMatrixStrided<T, device>());
}

template <class T, Device device>
void testAllReduce(comm::Communicator world, matrix::Matrix<T, device> matA,
                   matrix::Matrix<T, device> matB) {
  common::Pipeline<comm::Communicator> chain(world);

  const auto root_rank = world.size() - 1;
  const auto ex_mpi = dlaf::getMPIExecutor<Backend::MC>();

  const LocalTileIndex idx(0, 0);
  matrix::Matrix<T, device>& mat_in = root_rank % 2 == 0 ? matA : matB;
  matrix::Matrix<T, device>& mat_out = root_rank % 2 == 0 ? matB : matA;

  // set -> use -> read
  auto input_tile = fixedValueTile(world.rank() + 1);
  matrix::test::set(mat_in(idx).get(), input_tile);

  scheduleAllReduce(ex_mpi, chain(), MPI_SUM, mat_in.read(idx), mat_out(idx));

  CHECK_TILE_EQ(input_tile, mat_in.read(idx).get());

  auto exp_tile = fixedValueTile(world.size() * (world.size() + 1) / 2);
  CHECK_TILE_EQ(exp_tile, mat_out.read(idx).get());
}

TEST_F(CollectiveTest, AllReduce) {
  testAllReduce(world, newBlockMatrixContiguous<T, device>(), newBlockMatrixContiguous<T, device>());
  testAllReduce(world, newBlockMatrixStrided<T, device>(), newBlockMatrixStrided<T, device>());
  testAllReduce(world, newBlockMatrixContiguous<T, device>(), newBlockMatrixStrided<T, device>());
}

// TODO TEST AllReduce -> AllReduceInPlace Mixed
