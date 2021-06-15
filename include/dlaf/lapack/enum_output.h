//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <ostream>

#include <lapack/util.hh>

namespace lapack {

inline std::ostream& operator<<(std::ostream& out, const Norm& norm) {
  switch (norm) {
    case Norm::One:
      out << "OneNorm";
      break;
    case Norm::Two:
      out << "TwoNorm";
      break;
    case Norm::Inf:
      out << "InfinityNorm";
      break;
    case Norm::Fro:
      out << "FrobeniusNorm";
      break;
    case Norm::Max:
      out << "MaxNorm";
      break;
  }
  return out;
}

inline std::ostream& operator<<(std::ostream& out, const lapack::MatrixType& mtype) {
  switch (mtype) {
    case lapack::MatrixType::General:
      out << "General";
      break;
    case lapack::MatrixType::Lower:
      out << "Lower";
      break;
    case lapack::MatrixType::Upper:
      out << "Upper";
      break;
  }
  return out;
}

}
