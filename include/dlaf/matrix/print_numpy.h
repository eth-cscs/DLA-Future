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

#include <string>

#include "dlaf/common/index2d.h"
#include "dlaf/common/range2d.h"
#include "dlaf/matrix.h"
#include "dlaf/tile.h"
#include "dlaf/types.h"

namespace dlaf {
namespace matrix {

namespace internal {
template <class T>
constexpr const char* print_numpy_type(const T&) {
  return "np.float";
}

template <class T>
constexpr const char* print_numpy_type(const std::complex<T>&) {
  return "np.complex";
}

template <class T>
std::string print_numpy_value(const T& value) {
  return std::to_string(value);
}

template <class T>
std::string print_numpy_value(const std::complex<T>& value) {
  std::ostringstream os;
  os << "complex" << value;
  return os.str();
}
}

template <class Stream, class T, Device device, template <class, Device> class MatrixLikeT>
Stream& print_numpy(Stream& os, MatrixLikeT<const T, device>& matrix, std::string symbol) {
  using common::iterate_range2d;

  const auto& distribution = matrix.distribution();

  os << symbol << " = np.zeros(" << distribution.size() << ", dtype=" << internal::print_numpy_type(T{})
     << ")" << std::endl;

  for (const auto& index_tile : iterate_range2d(distribution.localNrTiles())) {
    const auto& tile = matrix.read(index_tile).get();

    for (const auto& index_el : iterate_range2d(tile.size())) {
      GlobalElementIndex index_g{
          distribution.template globalElementFromLocalTileAndTileElement<Coord::Row>(index_tile.row(),
                                                                                     index_el.row()),
          distribution.template globalElementFromLocalTileAndTileElement<Coord::Col>(index_tile.col(),
                                                                                     index_el.col()),
      };
      os << symbol << "[" << index_g.row() << "," << index_g.col()
         << "] = " << internal::print_numpy_value(tile(index_el)) << std::endl;
    }
  }

  return os;
}

template <class Stream, class T>
Stream& print_numpy(Stream& os, const dlaf::Tile<const T, Device::CPU>& tile) {
  os << "np.array([";
  for (const auto& index : iterate_range2d(tile.size()))
    os << internal::print_numpy_value(tile(index)) << ", ";
  os << "]).reshape" << tile.size();
  // since numpy reads a flat array as row-major, but iterate_range2d scans col-major
  os << ".T";
  return os;
}

}
}
