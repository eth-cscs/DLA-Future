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

#include <ostream>

#include <lapack_util.hh>

namespace lapack {

inline std::ostream& operator<<(std::ostream& out, const Norm& norm) {
  switch (norm) {
    case lapack::Norm::One:
      out << "One";
      break;
    case lapack::Norm::Two:
      out << "Two";
      break;
    case lapack::Norm::Inf:
      out << "Inf";
      break;
    case lapack::Norm::Fro:
      out << "Fro";
      break;
    case lapack::Norm::Max:
      out << "One";
      break;
  }
  return out;
}

}
