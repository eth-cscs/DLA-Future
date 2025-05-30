//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// @file

#include <cstddef>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <type_traits>

#include <dlaf/common/range2d.h>
#include <dlaf/communication/communicator.h>
#include <dlaf/matrix/distribution.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/matrix_ref.h>
#include <dlaf/util_math.h>

#include <gtest/gtest.h>

#include <dlaf_test/matrix/matrix_local.h>
#include <dlaf_test/matrix/util_tile.h>

namespace dlaf {
namespace matrix {
namespace test {

template <class MatrixType>
struct matrix_traits;

template <template <class, Device> class MatrixLike, class T, Device D>
struct matrix_traits<MatrixLike<T, D>> {
  using ElementT = std::remove_cv_t<T>;
  using has_distribution = std::true_type;
};

template <class T, Device D>
struct matrix_traits<Tile<T, D>> {
  using ElementT = std::remove_cv_t<T>;
  using has_distribution = std::false_type;
};

template <class T>
struct matrix_traits<MatrixLocal<T>> {
  using ElementT = std::remove_cv_t<T>;
  using has_distribution = std::false_type;
};

/// Sets the elements of the matrix.
///
/// The (i, j)-element of the matrix is set to el({i, j}).
/// @pre el argument is an index of type const GlobalElementIndex& or GlobalElementIndex,
/// @pre el return type should be T.
template <template <class, Device> class MatrixType, class T, class ElementGetter,
          class = std::enable_if_t<matrix_traits<MatrixType<T, Device::CPU>>::has_distribution::value>>
void set(MatrixType<T, Device::CPU>& mat, ElementGetter el) {
  const matrix::Distribution& dist = mat.distribution();
  for (SizeType tile_j = 0; tile_j < dist.localNrTiles().cols(); ++tile_j) {
    for (SizeType tile_i = 0; tile_i < dist.localNrTiles().rows(); ++tile_i) {
      auto tile_index = LocalTileIndex(tile_i, tile_j);
      auto tile_base_index =
          dist.globalElementIndex(dist.globalTileIndex(tile_index), TileElementIndex(0, 0));
      auto el_tile = [&el, &tile_base_index](const TileElementIndex& tile_index) {
        return el(GlobalElementIndex(tile_base_index.row() + tile_index.row(),
                                     tile_base_index.col() + tile_index.col()));
      };
      set(pika::this_thread::experimental::sync_wait(mat.readwrite(tile_index)), el_tile);
    }
  }
}

/// Returns an ElementGetter that given @p fullValues, it returns values like if origin has been changed
/// to sub-martix starting at @p offset.
template <class ElementGetter>
auto sub_values(ElementGetter&& fullValues, const GlobalElementIndex& offset) {
  return [fullValues, offset = sizeFromOrigin(offset)](const GlobalElementIndex& ij) {
    return fullValues(ij + offset);
  };
}

/// Returns an ElementGetter that returns values of a matrix like if:
/// - sub-matrix defined by @p sub_spec is set with @p insideValues
/// - the rest of the matrix is set with @p outsideValues
template <class OutsideElementGetter, class InsideElementGetter>
auto mix_values(const dlaf::matrix::internal::SubMatrixSpec& sub_spec,
                InsideElementGetter&& insideValues, OutsideElementGetter&& outsideValues) {
  return [outsideValues, insideValues, sub_spec](const GlobalElementIndex& ij) {
    if (ij.isInSub(sub_spec.origin, sub_spec.size))
      return insideValues(ij - common::sizeFromOrigin(sub_spec.origin));
    else
      return outsideValues(ij);
  };
}

/// Return source_rank_index along @t coord so that sub-matrix with origin at @p offset_in in @p dist_in
/// and the one in @p offset_out with @p blocksize_out will have the top-left tile allocated on the same rank.
///
/// @pre dist_in.block_size().get<coord>() == blocksize_out.get<coord>()
template <Coord coord>
comm::IndexT_MPI align_sub_rank_index(const Distribution& dist_in, const GlobalElementIndex& offset_in,
                                      const TileElementSize& blocksize_out,
                                      const GlobalElementIndex& offset_out) {
  const SizeType blocksize = blocksize_out.get<coord>();
  DLAF_ASSERT(dist_in.block_size().get<coord>() == blocksize, dist_in.block_size().get<coord>(),
              blocksize);

  const SizeType grid_size = dist_in.grid_size().get<coord>();

  // Note:
  // if the sub-matrix has an origin outside the parent matrix, any rank is ok, since the sub-matrix
  // cannot exist.
  if (!offset_in.isIn(dist_in.size()))
    return to_int(grid_size) / 2;

  const auto pos_mod = [](const auto& a, const auto& b) {
    const auto mod = a % b;
    return (mod >= 0) ? mod : (mod + b);
  };

  const comm::IndexT_MPI sub_rank = dist_in.rank_global_element<coord>(offset_in.get<coord>());
  const comm::IndexT_MPI offset_rank = to_int(offset_out.get<coord>() / blocksize);

  return pos_mod(sub_rank - offset_rank, to_int(grid_size));
}

/// Return source_rank_index so that sub-matrix with origin at @p offset_in in @p dist_in and the one in
/// @p offset_out with @p blocksize_out will have the top-left tile allocated on the same rank.
///
/// @pre dist_in.block_size() == blocksize_out
comm::Index2D align_sub_rank_index(const Distribution& dist_in, const GlobalElementIndex& offset_in,
                                   const TileElementSize& blocksize_out,
                                   const GlobalElementIndex& offset_out) {
  return {align_sub_rank_index<Coord::Row>(dist_in, offset_in, blocksize_out, offset_out),
          align_sub_rank_index<Coord::Col>(dist_in, offset_in, blocksize_out, offset_out)};
}

namespace internal {

/// Checks the elements of the matrix.
///
/// comp(expected({i, j}), (i, j)-element) is used to compare the elements.
/// err_message(expected({i, j}), (i, j)-element) is printed for the first element
/// that does not fulfill the comparison.
/// @pre expected argument is an index of type const GlobalElementIndex&,
/// @pre comp should have two arguments and return true if the comparison is fulfilled and false otherwise,
/// @pre err_message should have two arguments and return a string,
/// @pre expected return type should be the same as the type of the first argument of comp and of err_message,
/// @pre The second argument of comp should be either T, T& or const T&,
/// @pre The second argument of err_message should be either T, T& or const T&.
template <template <class, Device> class MatrixType, class T, class ElementGetter, class ComparisonOp,
          class ErrorMessageGetter>
void check(ElementGetter expected, MatrixType<T, Device::CPU>& mat, ComparisonOp comp,
           ErrorMessageGetter err_message, const char* file, const int line) {
  const matrix::Distribution& dist = mat.distribution();
  for (SizeType tile_j = 0; tile_j < dist.localNrTiles().cols(); ++tile_j) {
    for (SizeType tile_i = 0; tile_i < dist.localNrTiles().rows(); ++tile_i) {
      auto wrapper =
          pika::this_thread::experimental::sync_wait(mat.read(LocalTileIndex(tile_i, tile_j)));
      auto& tile = wrapper.get();
      for (SizeType jj = 0; jj < tile.size().cols(); ++jj) {
        SizeType j = dist.globalElementFromLocalTileAndTileElement<Coord::Col>(tile_j, jj);
        for (SizeType ii = 0; ii < tile.size().rows(); ++ii) {
          SizeType i = dist.globalElementFromLocalTileAndTileElement<Coord::Row>(tile_i, ii);
          if (!comp(expected({i, j}), tile({ii, jj}))) {
            ADD_FAILURE_AT(file, line)
                << "Error at index (" << i << ", " << j
                << "): " << err_message(expected({i, j}), tile({ii, jj})) << std::endl;
            return;
          }
        }
      }
    }
  }
}

/// Checks the elements of the matrix.
///
/// comp(expected({i, j}), (i, j)-element) is used to compare the elements.
/// err_message(expected({i, j}), (i, j)-element) is printed for the first element
/// that does not fulfill the comparison.
/// @pre expected argument is an index of type const GlobalElementIndex&,
/// @pre comp should have two arguments and return true if the comparison is fulfilled and false otherwise,
/// @pre err_message should have two arguments and return a string,
/// @pre expected return type should be the same as the type of the first argument of comp and of err_message,
/// @pre The second argument of comp should be either T, T& or const T&,
/// @pre The second argument of err_message should be either T, T& or const T&.
template <class T, class ElementGetter, class ComparisonOp, class ErrorMessageGetter>
void check(ElementGetter&& expected, MatrixLocal<const T>& mat, ComparisonOp comp,
           ErrorMessageGetter err_message, const char* file, const int line) {
  for (const auto& index : dlaf::common::iterate_range2d(mat.size())) {
    if (!comp(expected(index), mat(index))) {
      ADD_FAILURE_AT(file, line) << "Error at index (" << index
                                 << "): " << err_message(expected(index), mat(index)) << std::endl;
      return;
    }
  }
}
}

/// Checks the elements of the matrix (exact equality).
///
/// The (i, j)-element of the matrix is compared to exp_el({i, j}).
/// @pre exp_el argument is an index of type const GlobalElementIndex&,
/// @pre exp_el return type should be T.
template <class MatrixType, class ElementGetter>
void checkEQ(ElementGetter&& exp_el, MatrixType& mat, const char* file, const int line) {
  using T = typename matrix_traits<MatrixType>::ElementT;
  auto err_message = [](T expected, T value) {
    std::stringstream s;
    s << "expected " << expected << " == " << value;
    return s.str();
  };
  internal::check(exp_el, mat, std::equal_to<T>{}, err_message, file, line);
}
#define CHECK_MATRIX_EQ(exp_el, mat) ::dlaf::matrix::test::checkEQ(exp_el, mat, __FILE__, __LINE__)

/// Checks the pointers to the elements of the matrix.
///
/// The pointer to (i, j)-element of the matrix is compared to exp_ptr({i, j}).
/// @pre exp_ptr argument is an index of type const GlobalElementIndex&,
/// @pre exp_ptr return type should be T*.
template <class MatrixType, class PointerGetter>
void checkPtr(PointerGetter exp_ptr, MatrixType& mat, const char* file, const int line) {
  using T = typename std::pointer_traits<decltype(exp_ptr({}))>::element_type;
  auto comp = [](T* ptr, const T& value) { return ptr == &value; };
  auto err_message = [](T* expected, const T& value) {
    std::stringstream s;
    s << "expected " << expected << " == " << &value;
    return s.str();
  };
  internal::check(exp_ptr, mat, comp, err_message, file, line);
}
#define CHECK_MATRIX_PTR(exp_ptr, mat) ::dlaf::matrix::test::checkPtr(exp_ptr, mat, __FILE__, __LINE__)

/// Checks the elements of the matrix.
///
/// The (i, j)-element of the matrix is compared to expected({i, j}).
/// @pre expected argument is an index of type const GlobalElementIndex&,
/// @pre expected return type should be T,
/// @pre rel_err >= 0,
/// @pre abs_err >= 0,
/// @pre rel_err > 0 || abs_err > 0.
template <class MatrixType, class ElementGetter>
void checkNear(ElementGetter&& expected, MatrixType& mat,
               BaseType<typename matrix_traits<MatrixType>::ElementT> rel_err,
               BaseType<typename matrix_traits<MatrixType>::ElementT> abs_err, const char* file,
               const int line) {
  using T = typename matrix_traits<MatrixType>::ElementT;
  ASSERT_GE(rel_err, 0);
  ASSERT_GE(abs_err, 0);
  ASSERT_TRUE(rel_err > 0 || abs_err > 0);

  auto comp = [rel_err, abs_err](T expected, T value) {
    auto diff = std::abs(expected - value);
    auto abs_max = std::max(std::abs(expected), std::abs(value));

    return (diff < abs_err) || (diff / abs_max < rel_err);
  };
  auto err_message = [rel_err, abs_err](T expected, T value) {
    auto diff = std::abs(expected - value);
    auto abs_max = std::max(std::abs(expected), std::abs(value));

    std::stringstream s;
    s << "expected " << expected << " == " << value << " (Relative diff: " << diff / abs_max << " > "
      << rel_err << ", Absolute diff: " << diff << " > " << abs_err << ")";
    return s.str();
  };
  internal::check(expected, mat, comp, err_message, file, line);
}
#define CHECK_MATRIX_NEAR(expected, mat, rel_err, abs_err) \
  ::dlaf::matrix::test::checkNear(expected, mat, rel_err, abs_err, __FILE__, __LINE__)

template <class MatrixType>
void checkMatrixDistribution(const Distribution& distribution, const MatrixType& matrix) {
  ASSERT_EQ(distribution, matrix.distribution());

  EXPECT_EQ(distribution.size(), matrix.size());
  EXPECT_EQ(distribution.block_size(), matrix.block_size());
  EXPECT_EQ(distribution.tile_size(), matrix.tile_size());
  EXPECT_EQ(distribution.nr_tiles(), matrix.nr_tiles());
  EXPECT_EQ(distribution.rank_index(), matrix.rank_index());
  EXPECT_EQ(distribution.grid_size(), matrix.grid_size());

  for (SizeType j = 0; j < distribution.nrTiles().cols(); ++j) {
    for (SizeType i = 0; i < distribution.nrTiles().rows(); ++i) {
      GlobalTileIndex index(i, j);
      EXPECT_EQ(distribution.rank_global_tile(index), matrix.rank_global_tile(index));
    }
  }
}
#define CHECK_MATRIX_DISTRIBUTION(distribution, mat)                  \
  do {                                                                \
    std::stringstream s;                                              \
    s << "Rank " << mat.distribution().rankIndex();                   \
    SCOPED_TRACE(s.str());                                            \
    ::dlaf::matrix::test::checkMatrixDistribution(distribution, mat); \
  } while (0)

}
}
}
