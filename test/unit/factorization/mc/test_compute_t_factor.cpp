//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2020, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/factorization/qr.h"

#include <gtest/gtest.h>
#include <blas.hh>

#include "dlaf/common/range2d.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/lapack_tile.h"  // workaround for importing lapack.hh
#include "dlaf/matrix/matrix.h"
#include "dlaf/util_matrix.h"

#include "dlaf_test/comm_grids/grids_6_ranks.h"
#include "dlaf_test/matrix/matrix_local.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/matrix/util_matrix_local.h"
#include "dlaf_test/util_types.h"

using dlaf::Device;
using dlaf::Matrix;
using dlaf::comm::CommunicatorGrid;
using dlaf::matrix::test::MatrixLocal;

constexpr auto DEVICE_CPU = Device::CPU;

template <class T>
T el_eye(const dlaf::GlobalElementIndex& index) {
  return index.row() == index.col() ? 1 : 0;
}

template <class T>
void is_orthogonal(const MatrixLocal<const T>& matrix) {
  MatrixLocal<T> ortho(matrix.size(), matrix.blockSize());
  MatrixLocal<const T> eye = [&]() {
    MatrixLocal<T> m(matrix.size(), matrix.blockSize());
    set(m, el_eye<T>);
    return m;
  }();

  // beta = 1
  // alpha = -1
  // op(V) = NoTrans
  // op(W) = ConjTrans
  // I = beta I + alpha V W

  // clang-format off
  blas::gemm(blas::Layout::ColMajor,
      blas::Op::NoTrans, blas::Op::ConjTrans,
      matrix.size().rows(), matrix.size().cols(), matrix.size().rows(),
      1,
      matrix.ptr(), matrix.ld(),
      matrix.ptr(), matrix.ld(),
      0,
      ortho.ptr(), ortho.ld());
  // clang-format on

  // TODO fix this
  constexpr auto error = 1e-4;  // dlaf::test::TypeUtilities<T>::error;
  CHECK_MATRIX_NEAR(eye, ortho, error, error);
}

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new dlaf::test::CommunicatorGrid6RanksEnvironment);

template <typename Type>
struct ComputeTFactorDistributedTest : public ::testing::Test {
  const std::vector<dlaf::comm::CommunicatorGrid>& commGrids() {
    return dlaf::test::comm_grids;
  }
};

TYPED_TEST_SUITE(ComputeTFactorDistributedTest, dlaf::test::MatrixElementTypes);

std::vector<std::array<dlaf::SizeType, 5>> configs{
    // m, n, mb, nb, k
    {39, 26, 6, 6, 6},  // all reflectors
    {39, 26, 6, 6, 4},  // k reflectors
};

TYPED_TEST(ComputeTFactorDistributedTest, Correctness) {
  using namespace dlaf;

  // TODO fix this
  constexpr auto error = 1e-4;  // test::TypeUtilities<TypeParam>::error;

  for (auto comm_grid : this->commGrids()) {
    for (auto config : configs) {
      SizeType m, n, mb, nb, k;
      std::tie(m, n, mb, nb, k) = config;

      DLAF_ASSERT(k <= nb, k, nb, "reflectors should be in the same column of tiles");

      const GlobalElementSize size(m, n);
      const TileElementSize block_size(mb, nb);

      const matrix::Distribution distribution(size, block_size, comm_grid.size(), comm_grid.rank(),
                                              {0, 0});

      Matrix<const TypeParam, DEVICE_CPU> v_input = [&]() {
        Matrix<TypeParam, Device::CPU> V(distribution);
        dlaf::matrix::util::set_random(V);
        return V;
      }();

      const MatrixLocal<const TypeParam> v = [&]() {
        auto v = matrix::test::all_gather<TypeParam>(v_input, comm_grid);
        const auto min_size = std::min(m, n);
        lapack::laset(lapack::MatrixType::Upper, min_size, min_size, 0, 1, v.ptr({0, 0}), v.ld());
        return v;
      }();

      common::internal::vector<hpx::shared_future<TypeParam>> taus_input;
      taus_input.reserve(k);

      // compute taus and H_exp
      MatrixLocal<TypeParam> h_expected({m, m}, block_size);
      set(h_expected, el_eye<TypeParam>);

      for (auto i = 0; i < k; ++i) {
        const TypeParam* data_ptr = v.ptr({i, i});
        const auto norm = blas::nrm2(m - i, data_ptr, 1);
        const TypeParam tau = 2 / std::pow(norm, 2);

        taus_input.push_back(hpx::make_ready_future<TypeParam>(tau));

        // TODO work just on the submatrix

        MatrixLocal<TypeParam> h_i({m, m}, block_size);
        set(h_i, el_eye<TypeParam>);

        // Hi = (I - tau . v . v*)
        // clang-format off
        blas::ger(blas::Layout::ColMajor,
            m - i, m - i,
            -tau,
            v.ptr({i, i}), 1,
            v.ptr({i, i}), 1,
            h_i.ptr({i, i}), h_i.ld());
        // clang-format on

        // H_exp = H_exp * Hi
        // clang-format off
        MatrixLocal<TypeParam> workspace(h_expected.size(), h_expected.blockSize());
        blas::gemm(blas::Layout::ColMajor,
            blas::Op::NoTrans, blas::Op::NoTrans,
            m, m, m,
            1,
            h_expected.ptr(), h_expected.ld(),
            h_i.ptr(), h_i.ld(),
            0,
            workspace.ptr(), workspace.ld());
        // clang-format on

        // TODO improve this
        copy(workspace, h_expected);
      }

      is_orthogonal(h_expected);

      // TODO call the function to be tested
      Matrix<TypeParam, Device::CPU> t_output(LocalElementSize{k, k}, block_size);
      common::Pipeline<comm::CommunicatorGrid> serial_comm(comm_grid);

      // TODO just the first column panel is tested
      dlaf::factorization::internal::computeTFactor<Backend::MC>(k, v_input, {0, 0}, taus_input,
                                                                 t_output, serial_comm);

      // TODO T factor is reduced just on the rank owning V0
      const comm::Index2D owner_t{0, 0};
      if (distribution.rankIndex() != owner_t)
        continue;

      const auto& t = t_output.read(LocalTileIndex{0, 0}).get();

      // compute H_result = I - VTV*

      // TODO W = T V*
      MatrixLocal<TypeParam> w({m, k}, block_size);
      std::copy(v.ptr(), v.ptr() + w.size().linear_size(), w.ptr());

      // alpha = 1
      // op(T) = NoTrans
      // V = alpha op(T) . V
      // TV* = VT* = W
      // clang-format off
      blas::trmm(blas::Layout::ColMajor,
          blas::Side::Right, blas::Uplo::Upper,
          blas::Op::ConjTrans, blas::Diag::NonUnit,
          m, k,
          1,
          t.ptr(), t.ld(),
          w.ptr(), w.ld());
      // clang-format on

      // TODO H_result = I - V W
      MatrixLocal<TypeParam> h_result(h_expected.size(), block_size);
      set(h_result, el_eye<TypeParam>);

      // beta = 1
      // alpha = -1
      // op(V) = NoTrans
      // op(W) = ConjTrans
      // I = beta I + alpha V W

      // clang-format off
      blas::gemm(blas::Layout::ColMajor,
          blas::Op::NoTrans, blas::Op::ConjTrans,
          m, m, k,
          -1,
          v.ptr(), v.ld(),
          w.ptr(), w.ld(),
          1,
          h_result.ptr(), h_result.ld());
      // clang-format on

      is_orthogonal(h_result);

      // check H_result == H_exp
      CHECK_MATRIX_NEAR(h_expected, h_result, error, error);
    }
  }
}
