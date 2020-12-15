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

#include <string>

#include "dlaf/common/index2d.h"
#include "dlaf/common/range2d.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/matrix/panel.h"
#include "dlaf/types.h"

namespace dlaf {

namespace format {
struct numpy {};
}

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
void print(format::numpy, const Tile<const T, Device::CPU>& tile, std::ostream& os = std::cout) {
  os << "np.array([";

  // Note:
  // iterate_range2d loops over indices in column-major order, while python default
  // order is row-major.
  // For this reason, values are printed in a column-major order, and reordering is deferred
  // to python by tranposing the resulting array (and shaping it accordingly)
  for (const auto& index : iterate_range2d(tile.size()))
    os << internal::numpy_value(tile(index)) << ",";

  os << "]"
     << ", dtype=np." << internal::numpy_datatype<T>::typestring << ")"
     << ".reshape" << transposed(tile.size()) << ".T\n";
}

template <class T, Device device, template <class, Device> class MatrixLikeT>
void print(format::numpy, std::string sym, MatrixLikeT<const T, device>& matrix,
           std::ostream& os = std::cout) {
  using common::iterate_range2d;

  const auto& distribution = matrix.distribution();

  os << sym << " = np.zeros(" << distribution.size() << ", dtype=np."
     << internal::numpy_datatype<T>::typestring << ")\n";

  const LocalTileSize local_tiles = distribution.localNrTiles();

  auto getTileTopLeft = [&distribution](const LocalTileIndex& local) -> GlobalElementIndex {
    return {
        distribution.template globalElementFromLocalTileAndTileElement<Coord::Row>(local.row(), 0),
        distribution.template globalElementFromLocalTileAndTileElement<Coord::Col>(local.col(), 0),
    };
  };

  for (const auto& ij_local : iterate_range2d(local_tiles)) {
    const auto& tile = matrix.read(ij_local).get();

    const auto tl = getTileTopLeft(ij_local);
    const auto br = tl + GlobalElementSize{tile.size().rows(), tile.size().cols()};

    os << sym << "[" << tl.row() << ":" << br.row() << "," << tl.col() << ":" << br.col() << "] = ";
    print(format::numpy{}, tile, os);
  }
}

template <Coord panel_type, class T, Device device>
void print(format::numpy, std::string sym, Panel<panel_type, const T, device>& workspace,
           std::ostream& os = std::cout) {
  using common::iterate_range2d;

  const auto& distribution = workspace.distribution_matrix();

  os << sym << " = np.zeros(" << workspace.size() << ", dtype=np."
     << internal::numpy_datatype<T>::typestring << ")\n";

  auto getTileTopLeft = [&distribution](const LocalTileIndex& local) -> GlobalElementIndex {
    return {
        distribution.template globalElementFromLocalTileAndTileElement<Coord::Row>(local.row(), 0),
        distribution.template globalElementFromLocalTileAndTileElement<Coord::Col>(local.col(), 0),
    };
  };

  for (const auto& ij_local : iterate_range2d(workspace.offset(), workspace.localNrTiles())) {
    const auto& tile = workspace.read(ij_local).get();

    const auto tl = getTileTopLeft(ij_local);
    const auto br = tl + GlobalElementSize{tile.size().rows(), tile.size().cols()};

    os << sym << "[" << tl.row() << ":" << br.row() << "," << tl.col() << ":" << br.col() << "] = ";
    print(format::numpy{}, tile, os);
  }
}
}
}
