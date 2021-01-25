//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/communication/functions_sync.h"

#include <gtest/gtest.h>

#include "dlaf_test/helper_communicators.h"
#include "dlaf_test/matrix/util_tile.h"
#include "dlaf_test/util_types.h"

#include "dlaf/matrix/matrix.h"

using namespace dlaf;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace dlaf::test;
using namespace dlaf::comm;

using BroadcastMatrixTest = dlaf::comm::test::SplittedCommunicatorsTest;

template <class T>
T message_values(const TileElementIndex& index) {
  return TypeUtilities<T>::element((index.row() + 1) + (index.col() + 1) * 10, (index.col() + 1));
}

using TypeParam = std::complex<float>;

TEST_F(BroadcastMatrixTest, Matrix2Workspace) {
  if (splitted_comm.rank() == 0) {
    Matrix<TypeParam, Device::CPU> mat({26, 13}, {2, 2});

    LocalTileIndex selected_tile{0, 1};
    auto tile = mat(selected_tile).get();

    set(tile, message_values<TypeParam>);

    sync::broadcast::send(splitted_comm, tile);
  }
  else {
    TypeParam data[4];
    Tile<TypeParam, Device::CPU> workspace({2, 2}, {data, 4}, 2);

    set(workspace, TypeParam{});

    sync::broadcast::receive_from(0, splitted_comm, workspace);

    CHECK_TILE_EQ(message_values<TypeParam>, workspace);
  }
}

TEST_F(BroadcastMatrixTest, ConstMatrix2Workspace) {
  if (splitted_comm.rank() == 0) {
    Matrix<TypeParam, Device::CPU> mat({26, 13}, {2, 2});

    LocalTileIndex selected_tile{0, 1};
    set(mat(selected_tile).get(), message_values<TypeParam>);

    Matrix<const TypeParam, Device::CPU>& const_mat = mat;
    sync::broadcast::send(splitted_comm, const_mat.read(selected_tile).get());
  }
  else {
    TypeParam data[4];
    Tile<TypeParam, Device::CPU> workspace({2, 2}, {data, 4}, 2);

    set(workspace, TypeParam{});

    sync::broadcast::receive_from(0, splitted_comm, workspace);

    CHECK_TILE_EQ(message_values<TypeParam>, workspace);
  }
}

TEST_F(BroadcastMatrixTest, Matrix2Matrix) {
  if (splitted_comm.rank() == 0) {
    Matrix<TypeParam, Device::CPU> mat({5, 10}, {2, 2});

    LocalTileIndex selected_tile{0, 1};
    auto tile = mat(selected_tile).get();

    set(tile, message_values<TypeParam>);

    sync::broadcast::send(splitted_comm, tile);
  }
  else {
    Matrix<TypeParam, Device::CPU> mat({10, 5}, {2, 2});

    LocalTileIndex selected_tile{0, 1};
    auto tile = mat(selected_tile).get();

    set(tile, TypeParam{});

    sync::broadcast::receive_from(0, splitted_comm, tile);

    CHECK_TILE_EQ(message_values<TypeParam>, tile);
  }
}

TEST_F(BroadcastMatrixTest, ConstMatrix2Matrix) {
  if (splitted_comm.rank() == 0) {
    Matrix<TypeParam, Device::CPU> mat({5, 10}, {2, 2});

    LocalTileIndex selected_tile{0, 1};

    set(mat(selected_tile).get(), message_values<TypeParam>);

    Matrix<const TypeParam, Device::CPU>& const_mat = mat;
    sync::broadcast::send(splitted_comm, const_mat.read(selected_tile).get());
  }
  else {
    Matrix<TypeParam, Device::CPU> mat({10, 5}, {2, 2});

    LocalTileIndex selected_tile{0, 1};
    auto tile = mat(selected_tile).get();

    set(tile, TypeParam{});

    sync::broadcast::receive_from(0, splitted_comm, tile);

    CHECK_TILE_EQ(message_values<TypeParam>, tile);
  }
}

TEST_F(BroadcastMatrixTest, Workspace2Matrix) {
  if (splitted_comm.rank() == 0) {
    TypeParam data[4];
    Tile<TypeParam, Device::CPU> workspace({2, 2}, {data, 4}, 2);

    set(workspace, message_values<TypeParam>);

    sync::broadcast::send(splitted_comm, workspace);
  }
  else {
    Matrix<TypeParam, Device::CPU> mat({13, 26}, {2, 2});

    LocalTileIndex selected_tile{0, 1};
    auto tile = mat(selected_tile).get();

    set(tile, TypeParam{});

    sync::broadcast::receive_from(0, splitted_comm, tile);

    CHECK_TILE_EQ(message_values<TypeParam>, tile);
  }
}

TEST_F(BroadcastMatrixTest, ConstWorkspace2Matrix) {
  if (splitted_comm.rank() == 0) {
    TypeParam data[4];
    Tile<TypeParam, Device::CPU> workspace({2, 2}, {data, 4}, 2);

    set(workspace, message_values<TypeParam>);

    const Tile<const TypeParam, Device::CPU>& const_workspace = workspace;
    sync::broadcast::send(splitted_comm, const_workspace);
  }
  else {
    Matrix<TypeParam, Device::CPU> mat({13, 26}, {2, 2});

    LocalTileIndex selected_tile{0, 1};
    auto tile = mat(selected_tile).get();

    set(tile, TypeParam{});

    sync::broadcast::receive_from(0, splitted_comm, tile);

    CHECK_TILE_EQ(message_values<TypeParam>, tile);
  }
}

// it is the same as test_broadcast_tile
TEST_F(BroadcastMatrixTest, Workspace2Workspace) {
  if (splitted_comm.rank() == 0) {
    TypeParam data[4];
    Tile<TypeParam, Device::CPU> workspace({2, 2}, {data, 4}, 2);

    set(workspace, message_values<TypeParam>);

    sync::broadcast::send(splitted_comm, workspace);
  }
  else {
    TypeParam data[4];
    Tile<TypeParam, Device::CPU> workspace({2, 2}, {data, 4}, 2);

    set(workspace, TypeParam{});

    sync::broadcast::receive_from(0, splitted_comm, workspace);

    CHECK_TILE_EQ(message_values<TypeParam>, workspace);
  }
}

TEST_F(BroadcastMatrixTest, ConstWorkspace2Workspace) {
  if (splitted_comm.rank() == 0) {
    TypeParam data[4];
    Tile<TypeParam, Device::CPU> workspace({2, 2}, {data, 4}, 2);

    set(workspace, message_values<TypeParam>);

    const Tile<const TypeParam, Device::CPU>& const_workspace = workspace;
    sync::broadcast::send(splitted_comm, const_workspace);
  }
  else {
    TypeParam data[4];
    Tile<TypeParam, Device::CPU> workspace({2, 2}, {data, 4}, 2);

    set(workspace, TypeParam{});

    sync::broadcast::receive_from(0, splitted_comm, workspace);

    CHECK_TILE_EQ(message_values<TypeParam>, workspace);
  }
}
