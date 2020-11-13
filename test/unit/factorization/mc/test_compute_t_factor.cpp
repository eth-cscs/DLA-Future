//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/factorization/qr/t_factor_mc.h"

#include <gtest/gtest.h>
#include <blas.hh>

#include "dlaf/common/range2d.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/lapack_tile.h"  // workaround for importing lapack.hh
#include "dlaf/matrix.h"

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

  // CHECK_MATRIX_NEAR(eye, ortho, 1e-3, 1e-3);

  // std::cout << "O = ";
  // dlaf::matrix::print_numpy(std::cout, ortho);
  // std::cout << '\n';
}

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new dlaf::test::CommunicatorGrid6RanksEnvironment);

template <typename Type>
struct ComputeTFactorDistributedTest : public ::testing::Test {
  const std::vector<dlaf::comm::CommunicatorGrid>& commGrids() {
    return dlaf::test::comm_grids;
  }
};

using TestTheseTypes = ::testing::Types<float>;
TYPED_TEST_SUITE(ComputeTFactorDistributedTest, dlaf::test::MatrixElementTypes);

TYPED_TEST(ComputeTFactorDistributedTest, Correctness) {
  using namespace dlaf;

  for (auto comm_grid : this->commGrids()) {
    const SizeType m = 30;
    const SizeType n = 10;
    const SizeType nb = n;
    const SizeType k = nb;

    DLAF_ASSERT(k <= nb, k, nb, "you cannot have more reflectors than the block for now");

    const GlobalElementSize size(m, n);
    const TileElementSize block_size(nb, nb);

    const matrix::Distribution distribution({size.rows(), size.cols()}, block_size, comm_grid.size(),
                                            comm_grid.rank(), {0, 0});

    const auto rank = comm_grid.rank();
    const auto rank_col = 0;  // TODO generalize this

    if (rank.col() != rank_col) {
      std::cout << "NOT IN THE GAME\n";
      return;
    }

    Matrix<const TypeParam, DEVICE_CPU> V = [&]() {
      Matrix<TypeParam, Device::CPU> V(std::move(distribution));
      dlaf::matrix::util::set_random(V);
      return V;
    }();

    const MatrixLocal<const TypeParam> localV = [&]() {
      auto v = matrix::test::all_gather<TypeParam>(V, comm_grid);
      lapack::laset(lapack::MatrixType::Upper, n, n, 0, 1, v.ptr({0, 0}), v.ld());
      return v;
    }();

    common::internal::vector<hpx::shared_future<TypeParam>> taus;
    taus.reserve(k);

    // compute taus and H_exp
    MatrixLocal<TypeParam> H_exp({m, m}, block_size);
    set(H_exp, el_eye<TypeParam>);

    for (auto i = 0; i < k; ++i) {
      const TypeParam* data_ptr = localV.ptr({i, i});
      const auto norm = blas::nrm2(m - i, data_ptr, 1);
      const TypeParam tau = 2 / std::pow(norm, 2);

      taus.push_back(hpx::make_ready_future<TypeParam>(tau));

      // TODO work just on the submatrix

      MatrixLocal<TypeParam> Hi({m, m}, block_size);
      set(Hi, el_eye<TypeParam>);

      // Hi = (I - tau . v . v*)
      // clang-format off
      blas::ger(blas::Layout::ColMajor,
          m - i, m - i,
          -tau,
          localV.ptr({i, i}), 1,
          localV.ptr({i, i}), 1,
          Hi.ptr({i, i}), Hi.ld());
      // clang-format on

      // H_exp = H_exp * Hi
      // clang-format off
      MatrixLocal<TypeParam> workspace(H_exp.size(), H_exp.blockSize());
      blas::gemm(blas::Layout::ColMajor,
          blas::Op::NoTrans, blas::Op::NoTrans,
          m, m, m,
          1,
          H_exp.ptr(), H_exp.ld(),
          Hi.ptr(), Hi.ld(),
          0,
          workspace.ptr(), workspace.ld());
      // clang-format on

      // TODO improve this
      copy(workspace, H_exp);
    }

    is_orthogonal(H_exp);

    // TODO call the function to be tested
    Matrix<TypeParam, Device::CPU> T(LocalElementSize{k, k}, block_size);
    common::Pipeline<comm::CommunicatorGrid> serial_comm(comm_grid);

    dlaf::factorization::internal::QR<Backend::MC, Device::CPU>::computeTFactor(T, V, {0, 0}, {0, 0},
                                                                                k - 1, taus,
                                                                                serial_comm);

    // TODO compute H_result = I - VTV*
    // TODO W = T V*
    const auto localT = matrix::test::all_gather<const TypeParam>(T, comm_grid);

    MatrixLocal<TypeParam> W({m, n}, block_size);
    copy(localV, W);

    // alpha = 1
    // op(T) = NoTrans
    // V = alpha op(T) . V
    // TV* = VT* = W

    // clang-format off
    blas::trmm(blas::Layout::ColMajor,
        blas::Side::Right, blas::Uplo::Upper,
        blas::Op::ConjTrans, blas::Diag::NonUnit,
        m, n,
        1,
        localT.ptr(), localT.ld(),
        W.ptr(), W.ld());
    // clang-format on

    // TODO H_result = I - V W
    MatrixLocal<TypeParam> H_result({m, m}, block_size);
    set(H_result, el_eye<TypeParam>);

    // beta = 1
    // alpha = -1
    // op(V) = NoTrans
    // op(W) = ConjTrans
    // I = beta I + alpha V W

    // clang-format off
    blas::gemm(blas::Layout::ColMajor,
        blas::Op::NoTrans, blas::Op::ConjTrans,
        m, m, n,
        -1,
        localV.ptr(), localV.ld(),
        W.ptr(), W.ld(),
        1,
        H_result.ptr(), H_result.ld());
    // clang-format on

    is_orthogonal(H_result);

    // check H_result == H_exp
    // CHECK_MATRIX_NEAR(H_exp, H_result, 1e-3, 1e-3);
  }
}
