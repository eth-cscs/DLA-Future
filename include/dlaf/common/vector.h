//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <vector>

namespace dlaf {
namespace common {

/// std::vector with overloads for signed indexes
template <typename T>
struct vector : public std::vector<T> {
  vector(int size) : std::vector<T>(static_cast<std::size_t>(size)) {}

  T& operator[](int index) {
    return std::vector<T>::operator[](static_cast<std::size_t>(index));
  }

  const T& operator[](int index) const {
    return std::vector<T>::operator[](static_cast<std::size_t>(index));
  }
};

}
}
