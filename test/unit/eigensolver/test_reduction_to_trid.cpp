//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <dlaf/common/index2d.h>
#include <dlaf/common/range2d.h>
#include <dlaf/eigensolver/reduction_to_trid.h>
#include <dlaf/lapack/tile.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/matrix_mirror.h>
#include <dlaf/memory/memory_view.h>
#include <dlaf/types.h>
#include <dlaf/util_math.h>
#include <dlaf/util_matrix.h>

#include <gtest/gtest.h>

#include <dlaf_test/eigensolver/reduction_utils.h>
#include <dlaf_test/matrix/matrix_local.h>
#include <dlaf_test/matrix/util_matrix.h>
#include <dlaf_test/matrix/util_matrix_local.h>
#include <dlaf_test/matrix/util_tile.h>
#include <dlaf_test/util_types.h>

using namespace dlaf;
using namespace dlaf::test;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;

template <class T>
struct ReductionToTridTest : public testing::Test {};

template <class T>
using ReductionToTridTestMC = ReductionToTridTest<T>;

TYPED_TEST_SUITE(ReductionToTridTestMC, MatrixElementTypes);

#ifdef DLAF_WITH_GPU
template <class T>
using ReductionToTridTestGPU = ReductionToTridTest<T>;

TYPED_TEST_SUITE(ReductionToTridTestGPU, MatrixElementTypes);
#endif

namespace {

template <class T>
MatrixLocal<T> makeLocal(const Matrix<const T, Device::CPU>& matrix) {
  return {matrix.size(), matrix.distribution().baseTileSize()};
}

}

struct config_t {
  const SizeType m;
  const SizeType mb;
};

std::vector<config_t> configs{
    {0, 3},                                              // empty
    {1, 1}, {1, 3},                                      // single element
    {3, 3},                                              // single tile
    {4, 2}, {6, 2}, {12, 2}, {12, 3}, {24, 3}, {40, 5},  // multi-tile complete
    {8, 3}, {7, 3}, {13, 3}, {25, 3}, {39, 5}            // multi-tile incomplete
};

template <class T, Backend B, Device D>
void testReductionToTridLocal(const SizeType m, const SizeType mb) {
  namespace ex = pika::execution::experimental;
  namespace tt = pika::this_thread::experimental;

  const LocalElementSize size(m, m);
  const TileElementSize tile_size(mb, mb);

  const Distribution distribution({size.rows(), size.cols()}, tile_size);

  // setup the reference input matrix
  Matrix<const T, Device::CPU> reference = [size = size, block_size = tile_size]() {
    Matrix<T, Device::CPU> reference(size, block_size);
    matrix::util::set_random_hermitian(reference);
    return reference;
  }();

  Matrix<T, Device::CPU> mat_a_h(distribution);
  copy(reference, mat_a_h);

  eigensolver::internal::TridiagResult1Stage<T> res = [&]() mutable {
    MatrixMirror<T, D, Device::CPU> mat_a(mat_a_h);
    return eigensolver::internal::reduction_to_trid<B, D, T>(mat_a.get());
  }();

  ASSERT_EQ(res.taus.tile_size().rows(), tile_size.rows());

  dlaf::matrix::test::checkUpperPartUnchanged(reference, mat_a_h);

  auto mat_tri = makeLocal(mat_a_h);
  const auto& dist_tri = res.tridiagonal.distribution();
  for (SizeType i = 0; i < dist_tri.local_nr_tiles().rows(); ++i) {
    auto tile_tri_snd = tt::sync_wait(res.tridiagonal.read(LocalTileIndex{i, 0}));
    auto&& tile_tri = tile_tri_snd.get();
    for (SizeType i_el_tl = 0; i_el_tl < tile_tri.size().rows(); ++i_el_tl) {
      const SizeType i_el =
          dist_tri.template global_element_from_global_tile_and_tile_element<Coord::Row>(i, i_el_tl);

      mat_tri({i_el, i_el}) = tile_tri({i_el_tl, 0});
      if (i_el + 1 < mat_tri.size().rows())
        mat_tri({i_el + 1, i_el}) = mat_tri({i_el, i_el + 1}) = tile_tri({i_el_tl, 1});
    }
  }
  if (m > 2) {
    lapack::laset(lapack::Uplo::Lower, mat_tri.size().rows() - 2, mat_tri.size().cols() - 2, T{0}, T{0},
                  mat_tri.ptr({2, 0}), mat_tri.ld());
    lapack::laset(lapack::Uplo::Upper, mat_tri.size().rows() - 2, mat_tri.size().cols() - 2, T{0}, T{0},
                  mat_tri.ptr({0, 2}), mat_tri.ld());
  }

  auto mat_v = allGather(lapack::Uplo::Lower, mat_a_h);

  const auto k_reflectors = std::max<SizeType>(0, size.rows() - 1);
  auto taus = allGatherTaus(k_reflectors, res.taus);
  ASSERT_EQ(taus.size(), k_reflectors);

  checkResult(k_reflectors, 1, reference, mat_v, mat_tri, taus);
}

TYPED_TEST(ReductionToTridTestMC, CorrectnessLocal) {
  for (const auto& config : configs) {
    const auto& [m, mb] = config;
    testReductionToTridLocal<TypeParam, Backend::MC, Device::CPU>(m, mb);
  }
}

#ifdef DLAF_WITH_GPU
TYPED_TEST(ReductionToTridTestGPU, CorrectnessLocal) {
  for (const auto& config : configs) {
    const auto& [m, mb] = config;

    testReductionToTridLocal<TypeParam, Backend::GPU, Device::GPU>(m, mb);
  }
}

#endif
