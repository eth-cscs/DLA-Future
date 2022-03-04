//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <ostream>
#include "blas.hh"

#include <dlaf/common/format_short.h>

namespace blas {
inline std::ostream& operator<<(std::ostream& out, const blas::Diag& diag) {
  switch (diag) {
    case blas::Diag::Unit:
      out << "Unit";
      break;
    case blas::Diag::NonUnit:
      out << "NonUnit";
      break;
  }
  return out;
}

inline std::ostream& operator<<(std::ostream& out, const dlaf::internal::FormatShort<blas::Diag>& diag) {
  switch (diag.value) {
    case blas::Diag::Unit:
      out << "U";
      break;
    case blas::Diag::NonUnit:
      out << "N";
      break;
  }
  return out;
}

inline std::ostream& operator<<(std::ostream& out, const blas::Layout& layout) {
  switch (layout) {
    case blas::Layout::RowMajor:
      out << "RowMajor";
      break;
    case blas::Layout::ColMajor:
      out << "ColMajor";
      break;
  }
  return out;
}

inline std::ostream& operator<<(std::ostream& out,
                                const dlaf::internal::FormatShort<blas::Layout>& layout) {
  switch (layout.value) {
    case blas::Layout::RowMajor:
      out << "R";
      break;
    case blas::Layout::ColMajor:
      out << "C";
      break;
  }
  return out;
}

inline std::ostream& operator<<(std::ostream& out, const blas::Op& trans) {
  switch (trans) {
    case blas::Op::NoTrans:
      out << "NoTrans";
      break;
    case blas::Op::Trans:
      out << "Trans";
      break;
    case blas::Op::ConjTrans:
      out << "ConjTrans";
      break;
  }
  return out;
}

inline std::ostream& operator<<(std::ostream& out, const dlaf::internal::FormatShort<blas::Op>& trans) {
  switch (trans.value) {
    case blas::Op::NoTrans:
      out << "N";
      break;
    case blas::Op::Trans:
      out << "T";
      break;
    case blas::Op::ConjTrans:
      out << "C";
      break;
  }
  return out;
}

inline std::ostream& operator<<(std::ostream& out, const blas::Side& side) {
  switch (side) {
    case blas::Side::Left:
      out << "Left";
      break;
    case blas::Side::Right:
      out << "Right";
      break;
  }
  return out;
}

inline std::ostream& operator<<(std::ostream& out, const dlaf::internal::FormatShort<blas::Side>& side) {
  switch (side.value) {
    case blas::Side::Left:
      out << "L";
      break;
    case blas::Side::Right:
      out << "R";
      break;
  }
  return out;
}

inline std::ostream& operator<<(std::ostream& out, const blas::Uplo& uplo) {
  switch (uplo) {
    case blas::Uplo::Lower:
      out << "Lower";
      break;
    case blas::Uplo::Upper:
      out << "Upper";
      break;
    case blas::Uplo::General:
      out << "General";
      break;
  }
  return out;
}

inline std::ostream& operator<<(std::ostream& out, const dlaf::internal::FormatShort<blas::Uplo>& uplo) {
  switch (uplo.value) {
    case blas::Uplo::Lower:
      out << "L";
      break;
    case blas::Uplo::Upper:
      out << "U";
      break;
    case blas::Uplo::General:
      out << "G";
      break;
  }
  return out;
}
}
