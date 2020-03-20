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
#include "blas.hh"

namespace blas {
std::ostream& operator<<(std::ostream& stream, const blas::Diag& diag) {
  switch (diag) {
    case blas::Diag::Unit:
      stream << "Unit";
      break;
    case blas::Diag::NonUnit:
      stream << "NonUnit";
  }
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const blas::Side& side) {
  switch (side) {
    case blas::Side::Left:
      stream << "Left";
      break;
    case blas::Side::Right:
      stream << "Right";
      break;
  }
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const blas::Op& trans) {
  switch (trans) {
    case blas::Op::NoTrans:
      stream << "NoTrans";
      break;
    case blas::Op::Trans:
      stream << "Trans";
      break;
    case blas::Op::ConjTrans:
      stream << "ConjTrans";
      break;
  }
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const blas::Uplo& uplo) {
  switch (uplo) {
    case blas::Uplo::Lower:
      stream << "Lower";
      break;
    case blas::Uplo::Upper:
      stream << "Upper";
      break;
    case blas::Uplo::General:
      stream << "General";
      break;
  }
  return stream;
}
}
