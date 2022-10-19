//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
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
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/memory/memory_view.h"

#include "dlaf_test/matrix/util_tile.h"

using namespace dlaf;
using namespace dlaf::matrix::test;
namespace ex = pika::execution::experimental;
namespace tt = pika::this_thread::experimental;

class CollectiveTest : public ::testing::Test {
  static_assert(NUM_MPI_RANKS >= 2, "at least 2 ranks are required");

protected:
  using T = int;
  static constexpr Device device = Device::CPU;

  comm::Communicator world = MPI_COMM_WORLD;
};

template <class T, Device D>
auto newBlockMatrixContiguous() {
  auto layout = matrix::colMajorLayout({13, 13}, {13, 13}, 13);
  auto dist = matrix::Distribution({13, 13}, {13, 13});

  auto matrix = matrix::Matrix<T, D>(dist, layout);

  EXPECT_TRUE(data_iscontiguous(common::make_data(matrix.read(LocalTileIndex(0, 0)).get())));

  return matrix;
}

template <class T, Device D>
auto newBlockMatrixStrided() {
  auto layout = matrix::colMajorLayout({13, 13}, {13, 13}, 26);
  auto dist = matrix::Distribution({13, 13}, {13, 13});

  auto matrix = matrix::Matrix<T, D>(dist, layout);

  EXPECT_FALSE(data_iscontiguous(common::make_data(matrix.read(LocalTileIndex(0, 0)).get())));

  return matrix;
}

template <class T, Device D>
void testReduceInPlace(comm::Communicator world, matrix::Matrix<T, D> matrix, std::string test_name) {
  common::Pipeline<comm::Communicator> chain(world);

  const auto root_rank = world.size() - 1;
  const LocalTileIndex idx(0, 0);

  auto input_tile = fixedValueTile(world.rank() + 1);
  matrix::test::set(matrix(idx).get(), input_tile);

  std::function<T(TileElementIndex)> exp_tile;
  if (root_rank == world.rank()) {
    // use -> read
    ex::start_detached(
        dlaf::comm::scheduleReduceRecvInPlace(chain(), MPI_SUM,
                                              ex::make_unique_any_sender(matrix.readwrite_sender(idx))));

    exp_tile = fixedValueTile(world.size() * (world.size() + 1) / 2);
  }
  else {
    // use -> read -> set -> read
    ex::start_detached(
        dlaf::comm::scheduleReduceSend(chain(), root_rank, MPI_SUM,
                                       ex::make_unique_any_sender(matrix.read_sender(idx))));

    CHECK_TILE_EQ(input_tile, matrix.read(idx).get());

    auto new_tile = fixedValueTile(26);
    matrix::test::set(matrix(idx).get(), new_tile);

    exp_tile = new_tile;
  }

  const auto& tile = matrix.read(idx).get();
  SCOPED_TRACE(test_name);

  CHECK_TILE_EQ(exp_tile, tile);
}

TEST_F(CollectiveTest, ReduceInPlace) {
  testReduceInPlace(world, newBlockMatrixContiguous<T, device>(), "Contiguous");
  testReduceInPlace(world, newBlockMatrixStrided<T, device>(), "Strided");
}

template <class T, Device D>
void testAllReduceInPlace(comm::Communicator world, matrix::Matrix<T, D> matrix, std::string test_name) {
  common::Pipeline<comm::Communicator> chain(world);

  const LocalTileIndex idx(0, 0);

  // set -> use -> read
  auto input_tile = fixedValueTile(world.rank() + 1);
  matrix::test::set(matrix(idx).get(), input_tile);

  auto after =
      dlaf::comm::scheduleAllReduceInPlace(chain(), MPI_SUM,
                                           ex::make_unique_any_sender(matrix.readwrite_sender(idx)));

  // Note:
  // The call `sync_wait(after)` waits for any scheduled task with the aim to ensure that no other task
  // will yield after it, so `SCOPED_TRACE` can be called safely.
  //
  // Moreover, the code block is needed in order to limit the lifetime of `tile`, so that just after
  // it, it is possible to check the read operation (which implicitly depends on it)
  auto exp_tile = fixedValueTile(world.size() * (world.size() + 1) / 2);
  {
    auto tile = tt::sync_wait(std::move(after));
    SCOPED_TRACE(test_name);

    CHECK_TILE_EQ(exp_tile, tile);
  }

  CHECK_TILE_EQ(exp_tile, matrix.read(idx).get());
}

TEST_F(CollectiveTest, AllReduceInPlace) {
  testAllReduceInPlace(world, newBlockMatrixContiguous<T, device>(), "Contiguous");
  testAllReduceInPlace(world, newBlockMatrixStrided<T, device>(), "Strided");
}

template <class T, Device D>
void testAllReduce(comm::Communicator world, matrix::Matrix<T, D> matA, matrix::Matrix<T, D> matB,
                   std::string test_name) {
  common::Pipeline<comm::Communicator> chain(world);

  const auto root_rank = world.size() - 1;

  const LocalTileIndex idx(0, 0);
  matrix::Matrix<T, D>& mat_in = root_rank % 2 == 0 ? matA : matB;
  matrix::Matrix<T, D>& mat_out = root_rank % 2 == 0 ? matB : matA;

  // set -> use -> read
  auto input_tile = fixedValueTile(world.rank() + 1);
  matrix::test::set(mat_in(idx).get(), input_tile);

  ex::start_detached(
      dlaf::comm::scheduleAllReduce(chain(), MPI_SUM,
                                    ex::make_unique_any_sender(mat_in.read_sender(idx)),
                                    ex::make_unique_any_sender(mat_out.readwrite_sender(idx))));

  const auto& tile_in = mat_in.read(idx).get();
  const auto& tile_out = mat_out.read(idx).get();
  SCOPED_TRACE(test_name);

  CHECK_TILE_EQ(input_tile, tile_in);

  auto exp_tile = fixedValueTile(world.size() * (world.size() + 1) / 2);
  CHECK_TILE_EQ(exp_tile, tile_out);
}

TEST_F(CollectiveTest, AllReduce) {
  testAllReduce(world, newBlockMatrixContiguous<T, device>(), newBlockMatrixContiguous<T, device>(),
                "Contiguous<->Contiguous");
  testAllReduce(world, newBlockMatrixStrided<T, device>(), newBlockMatrixStrided<T, device>(),
                "Strided<->Strided");
  testAllReduce(world, newBlockMatrixContiguous<T, device>(), newBlockMatrixStrided<T, device>(),
                "Contiguous<->Strided");
}
