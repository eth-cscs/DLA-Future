//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause

#include "dlaf/eigensolver/backtransformation.h"

#include <gtest/gtest.h>

#include "dlaf/matrix/matrix.h"
#include "dlaf/util_matrix.h"

#include "dlaf_test/matrix/matrix_local.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/matrix/util_matrix_local.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::matrix;
using namespace dlaf::matrix::util;

using namespace dlaf::test;
using namespace dlaf::matrix::test;

template <typename Type>
class BacktransformationT2BTest : public ::testing::Test {};

TYPED_TEST_SUITE(BacktransformationT2BTest, MatrixElementTypes);

struct calculateTau {
  template <class T>
  static T call(const T* v, const SizeType size) {
    const T dotprod = blas::dot(size, v, 1, v, 1);
    return 2 / dotprod;
  }

  template <class T>
  static std::complex<T> call(const std::complex<T>* v, const SizeType size) {
    const T dotprod = std::real(blas::dot(size, v, 1, v, 1));
    const T imag = T(1) / size;  // TODO check tau vs dotprod
    return {(T(1.0) + std::sqrt(T(1) - dotprod * imag * imag)) / dotprod, imag};
  }

  // TODO random number has to be "resetted" at each time
};

template <class T>
void computeTaus(const SizeType n, const SizeType k, matrix::Tile<T, Device::CPU> tile) {
  for (SizeType j = 0; j < k; ++j) {
    SizeType size = std::min(n - j, tile.size().rows() - 1);
    DLAF_ASSERT(size > 0, size);
    const auto tau = calculateTau::call(tile.ptr({0, j}), size);
    *tile.ptr({0, j}) = tau;
  }
}

struct config_t {
  const SizeType m, n, mb, nb;
};

std::vector<config_t> configs{
    {0, 0, 4, 4},
    {12, 12, 4, 4},
    {12, 12, 4, 3},
    {10, 10, 3, 3},
    {8, 8, 3, 3},
    {20, 30, 5, 5},
    {20, 30, 5, 6},
    {12, 12, 5, 5},
    {12, 30, 5, 6},
};

TYPED_TEST(BacktransformationT2BTest, CorrectnessLocal) {
  struct algorithmConfig {
    algorithmConfig(SizeType m, SizeType mb) : m_(m), mb_(mb) {}

    SizeType nrSweeps() {
      return std::max<SizeType>(0, is_complex ? m_ - 1 : m_ - 2);
    }

    SizeType nrStepsPerSweep(SizeType sweep) {
      return std::max<SizeType>(0, sweep == m_ - 2 ? m_ - 1 : dlaf::util::ceilDiv(m_ - sweep - 2, mb_));
    }

    auto unzipReflector(const GlobalElementIndex ij) {
      const auto size = std::min(mb_, m_ - ij.row());
      const auto k = ij.col() % mb_;
      const auto i_t = (ij.row() - k + 1) / mb_;
      const auto j_t = ij.col() / mb_;
      return std::make_tuple(GlobalTileIndex(i_t, j_t), k, size);
    };

    const bool is_complex = std::is_same<TypeParam, ComplexType<TypeParam>>::value;
    SizeType m_, mb_;
  };

  for (const auto& config : configs) {
    const SizeType m = config.m;
    const SizeType n = config.n;
    const SizeType mb = config.mb;
    const SizeType nb = config.nb;

    algorithmConfig algConf(m, mb);

    //const SizeType b = mb;

    const LocalElementSize sz_e(m, n);
    const TileElementSize bsz_e(mb, nb);

    const LocalElementSize sz_v(m, m);
    const TileElementSize bsz_v(mb, mb);

    Matrix<TypeParam, Device::CPU> mat_e(sz_e, bsz_e);
    set_random(mat_e);
    auto mat_e_local = allGather(lapack::MatrixType::General, mat_e);

    Matrix<const TypeParam, Device::CPU> mat_v = [sz_v, bsz_v]() {
      Matrix<TypeParam, Device::CPU> mat_v(sz_v, bsz_v);
      set_random(mat_v);  // TODO ? same seed ==> mat_v == mat_e

      const auto m = mat_v.distribution().localNrTiles().cols();
      for (SizeType j = 0; j < m; ++j) {
        for (SizeType i = j; i < m; ++i) {
          const bool affectsTwoRows = i < m - 1;
          const SizeType k = affectsTwoRows ? mat_v.tileSize({i, j}).cols() : mat_v.tileSize({i, j}).rows() - 2;
          const SizeType n = mat_v.tileSize({i, j}).rows() - 1 + (affectsTwoRows ? mat_v.tileSize({i + 1, j}).rows() : 0);
          if (k <= 0)
            continue;
          hpx::dataflow(hpx::unwrapping(computeTaus<TypeParam>), n, k, mat_v(LocalTileIndex(i, j)));
        }
      }

      return mat_v;
    }();

    MatrixLocal<TypeParam> mat_v_local = allGather(lapack::MatrixType::Lower, mat_v);

    //std::cout << m << " " << n << " " << mb << " " << nb << std::endl;
    //print(format::numpy{}, "E", mat_e);
    //print(format::numpy{}, "V", mat_v);

    eigensolver::backTransformationT2B<Backend::MC>(mat_e, mat_v);

    for (SizeType sweep = algConf.nrSweeps() - 1; sweep >= 0; --sweep) {
      for (SizeType step = algConf.nrStepsPerSweep(sweep) - 1; step >= 0; --step) {
        const SizeType j = sweep;
        const SizeType i = j + 1 + step * mb;
        //std::cout << "i=" << i << "\nj=" << j << "\n";

        auto reflector = algConf.unzipReflector({i, j});

        const SizeType k = std::get<1>(reflector);
        const SizeType size = std::get<2>(reflector);

        //std::cout << "k=" << k << " size=" << size << "\n";

        auto& tile_v = mat_v_local.tile(std::get<0>(reflector));
        TypeParam& v_head = *tile_v.ptr({0, k});
        const TypeParam tau = v_head;
        v_head = 1;

        std::cout << "tau = " << tau << " " << "\tv = ";
        for (auto i = 0; i < size; ++i)
          std::cout << *tile_v.ptr({i, k}) << ", ";
        std::cout << "\n";

        lapack::larf(blas::Side::Left, size, n, &v_head, 1, tau, mat_e_local.ptr({i, 0}),
                     mat_e_local.ld());
      }
    }

    auto result = [& dist = mat_e.distribution(),
                   &mat_local = mat_e_local](const GlobalElementIndex& element) {
      const auto tile_index = dist.globalTileIndex(element);
      const auto tile_element = dist.tileElementIndex(element);
      return mat_local.tile_read(tile_index)(tile_element);
    };

    const auto error =
        std::max<SizeType>(1, 40 * m * n) * TypeUtilities<TypeParam>::error;  // TODO how much error
    CHECK_MATRIX_NEAR(result, mat_e, error, error);

    //print(format::numpy{}, "Etest", mat_e_local);
    //print(format::numpy{}, "Erun", mat_e);
  }
}
