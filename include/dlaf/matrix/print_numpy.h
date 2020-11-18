//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2020, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <string>

#include "dlaf/common/index2d.h"
#include "dlaf/common/range2d.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/types.h"

namespace dlaf {
namespace matrix {

namespace internal {

// For reference see https://numpy.org/doc/stable/user/basics.types.html
// Moreover, considering the small number of digits in the output,
// just single-precision types are used.
template <class T>
struct numpy_datatype {
  static constexpr auto typestring = "single";
};

template <class T>
struct numpy_datatype<std::complex<T>> {
  static constexpr auto typestring = "csingle";
};

template <class T>
std::string numpy_value(const T& value) {
  std::ostringstream os;
  os << value;
  return os.str();
}

template <class T>
std::string numpy_value(const std::complex<T>& value) {
  std::ostringstream os;
  const auto imag = std::imag(value);
  const auto sign = (imag < 0) ? '-' : '+';
  os << std::real(value) << sign << std::abs(imag) << "j";
  return os.str();
}
}

/// Print a tile as a numpy array
template <class T>
std::ostream& print_numpy(std::ostream& os, const Tile<const T, Device::CPU>& tile) {
  os << "np.array([";

  // Note:
  // iterate_range2d loops over indices in column-major order, while python default
  // order is row-major.
  // For this reason, values are printed in a column-major order, and reordering is deferred
  // to python by tranposing the resulting array (and shaping it accordingly)
  for (const auto& index : iterate_range2d(tile.size()))
    os << internal::numpy_value(tile(index)) << ", ";

  os << "]"
     << ", dtype=np." << internal::numpy_datatype<T>::typestring << ")"
     << ".reshape" << transposed(tile.size()) << ".T\n";

  return os;
}

template <class T, Device device, template <class, Device> class MatrixLikeT>
std::ostream& print_numpy(std::ostream& os, MatrixLikeT<const T, device>& matrix, std::string symbol) {
  using common::iterate_range2d;

  const auto& distribution = matrix.distribution();

  // clang-format off
  os
    << symbol << " = np.zeros(" << distribution.size()
    << ", dtype=np." << internal::numpy_datatype<T>::typestring << ")\n";
  // clang-format on

  const LocalTileSize local_tiles = distribution.localNrTiles();

  auto getTileTopLeft = [&distribution](const LocalTileIndex& local) -> GlobalElementIndex {
    return {
        distribution.template globalElementFromLocalTileAndTileElement<Coord::Row>(local.row(), 0),
        distribution.template globalElementFromLocalTileAndTileElement<Coord::Col>(local.col(), 0),
    };
  };

  for (const auto& ij_local : iterate_range2d(local_tiles)) {
    const auto& tile = matrix.read(ij_local).get();

    const auto index_tl = getTileTopLeft(ij_local);

    // clang-format off
    os
      << symbol << "["
      << index_tl.row() << ":" << index_tl.row() + tile.size().rows() << ", "
      << index_tl.col() << ":" << index_tl.col() + tile.size().cols()
      << "] = ";
    // clang-format on

    print_numpy(os, tile);
  }

  return os;
}

}
}
