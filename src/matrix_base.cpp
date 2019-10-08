//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/matrix_base.h"

namespace dlaf {
MatrixBase::MatrixBase() noexcept : size_(0, 0), nr_tiles_(0, 0), block_size_(1, 1) {}

MatrixBase::MatrixBase(const GlobalElementSize& size, const TileElementSize& block_size)
    : size_(size), nr_tiles_(0, 0), block_size_(block_size) {
  if (!size_.isValid())
    throw std::invalid_argument("Error: Invalid Matrix size");
  if (!block_size_.isValid() || block_size_.isEmpty())
    throw std::invalid_argument("Error: Invalid Block size");

  nr_tiles_ = {util::ceilDiv(size.rows(), block_size.rows()),
               util::ceilDiv(size.cols(), block_size.cols())};
}

MatrixBase::MatrixBase(const matrix::LayoutInfo& layout) noexcept
    : size_(layout.size()), nr_tiles_(layout.nrTiles()), block_size_(layout.blockSize()) {}

MatrixBase::MatrixBase(MatrixBase&& rhs) noexcept
    : size_(rhs.size_), nr_tiles_(rhs.nr_tiles_), block_size_(rhs.block_size_) {
  rhs.size_ = {0, 0};
  rhs.nr_tiles_ = {0, 0};
  rhs.block_size_ = {1, 1};
}

MatrixBase& MatrixBase::operator=(MatrixBase&& rhs) noexcept {
  size_ = rhs.size_;
  nr_tiles_ = rhs.nr_tiles_;
  block_size_ = rhs.block_size_;

  rhs.size_ = {0, 0};
  rhs.nr_tiles_ = {0, 0};
  rhs.block_size_ = {1, 1};

  return *this;
}
}
