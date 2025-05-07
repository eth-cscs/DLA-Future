//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// @file

#include <dlaf/common/index2d.h>
#include <dlaf/types.h>

namespace dlaf {
namespace matrix {
struct GlobalElement_TAG;
struct LocalElement_TAG;
struct GlobalTile_TAG;
struct LocalTile_TAG;
struct GlobalBlock_TAG;
struct LocalBlock_TAG;
struct TileElement_TAG;
}

using GlobalElementIndex = common::Index2D<SizeType, matrix::GlobalElement_TAG>;
using GlobalElementSize = common::Size2D<SizeType, matrix::GlobalElement_TAG>;

using LocalElementSize = common::Size2D<SizeType, matrix::LocalElement_TAG>;

using GlobalTileIndex = common::Index2D<SizeType, matrix::GlobalTile_TAG>;
using GlobalTileSize = common::Size2D<SizeType, matrix::GlobalTile_TAG>;

using LocalTileIndex = common::Index2D<SizeType, matrix::LocalTile_TAG>;
using LocalTileSize = common::Size2D<SizeType, matrix::LocalTile_TAG>;

using GlobalBlockIndex = common::Index2D<SizeType, matrix::GlobalBlock_TAG>;
using GlobalBlockSize = common::Size2D<SizeType, matrix::GlobalBlock_TAG>;

using LocalBlockIndex = common::Index2D<SizeType, matrix::LocalBlock_TAG>;
using LocalBlockSize = common::Size2D<SizeType, matrix::LocalBlock_TAG>;

using TileElementIndex = common::Index2D<SizeType, matrix::TileElement_TAG>;
using TileElementSize = common::Size2D<SizeType, matrix::TileElement_TAG>;
}
