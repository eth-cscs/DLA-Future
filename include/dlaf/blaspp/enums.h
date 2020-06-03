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

/// @file

#include <iostream>
#include <string>

#include <blas.hh>

namespace blas {

inline std::ostream& operator<<(std::ostream& out, Layout layout) noexcept {
  return out << layout2str(layout);
}

inline std::ostream& operator<<(std::ostream& out, Op op) noexcept {
  return out << op2str(op);
}

inline std::ostream& operator<<(std::ostream& out, Uplo uplo) noexcept {
  return out << uplo2str(uplo);
}

inline std::ostream& operator<<(std::ostream& out, Diag diag) noexcept {
  return out << diag2str(diag);
}

inline std::ostream& operator<<(std::ostream& out, Side side) noexcept {
  return out << side2str(side);
}

}
