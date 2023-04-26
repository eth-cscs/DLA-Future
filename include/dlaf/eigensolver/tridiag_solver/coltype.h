//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <iostream>

namespace dlaf::eigensolver::internal {

// The type of a column in the Q matrix
enum class ColType {
  UpperHalf,  // non-zeroes in the upper half only
  LowerHalf,  // non-zeroes in the lower half only
  Dense,      // full column vector
  Deflated    // deflated vectors
};

inline std::ostream& operator<<(std::ostream& str, const ColType& ct) {
  if (ct == ColType::Deflated) {
    str << "Deflated";
  }
  else if (ct == ColType::Dense) {
    str << "Dense";
  }
  else if (ct == ColType::UpperHalf) {
    str << "UpperHalf";
  }
  else {
    str << "LowerHalf";
  }
  return str;
}

}
