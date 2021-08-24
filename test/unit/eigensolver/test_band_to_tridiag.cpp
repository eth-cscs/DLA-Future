//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#include "dlaf/eigensolver/band_to_tridiag.h"

#include <exception>
#include <functional>
#include <sstream>
#include <tuple>
#include "gtest/gtest.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf_test/matrix/matrix_local.h"
#include "dlaf_test/matrix/util_generic_lapack.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/matrix/util_matrix_blas.h"
#include "dlaf_test/matrix/util_matrix_local.h"
#include "dlaf_test/util_types.h"

#include "dlaf/matrix/print_csv.h"
#include "dlaf/matrix/print_numpy.h"

using namespace dlaf;
using namespace dlaf::comm;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace dlaf::test;
using namespace testing;

TYPED_TEST_SUITE(EigensolverBandToTridiagTest, MatrixElementTypes);

const std::vector<std::tuple<SizeType, SizeType, SizeType>> sizes = {
    //{0, 2},                              // m = 0
    {5, 5, 5}, /*{4, 4, 2},*/              // m = mb
    /*{4, 6, 3},*/ {8, 4, 2},  /*{18, 4, 4}, {34, 6, 6}, {37, 9, 3}*/  // m != mb
};

template <class T>
void testBandToTridiag(const blas::Uplo uplo, const SizeType band_size, const SizeType m,
                       const SizeType mb) {
  const LocalElementSize size(m, m);
  const TileElementSize block_size(mb, mb);

  Matrix<T, Device::CPU> mat_a(size, block_size);

  auto el_a = [](auto index) { return T((double) 100 + 100 * index.row() - 99 * index.col()); };
  set(mat_a, el_a);
  matrix::print(format::numpy(), "A", mat_a);

  auto mat_trid = eigensolver::bandToTridiag<Backend::MC>(uplo, band_size, mat_a);
  matrix::print(format::csv(), "T", mat_trid);

  auto mat_local = matrix::test::allGather(lapack::char2matrixtype(blas::uplo2char(uplo)), mat_a);
  auto tmp = m - band_size - 1;
  if (tmp > 0) {
    if (uplo == blas::Uplo::Lower) {
      lapack::laset(lapack::MatrixType::Lower, tmp, tmp, 0., 0., mat_local.ptr({band_size + 1, 0}),
                    mat_local.ld());
    }
    else {
      DLAF_UNIMPLEMENTED(uplo);
    }
  }

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < m; ++j) {
      std::cout << *mat_local.ptr({i, j}) << ", ";
    }
    std::cout << std::endl;
  }
  common::internal::vector<BaseType<T>> e(m), d(m);
  common::internal::vector<T> tau(m);
  lapack::hetrd(uplo, m, mat_local.ptr(), mat_local.ld(), d.data(), e.data(), tau.data());

  for (auto el : d) {
    std::cout << el << ", ";
  }
  std::cout << std::endl;
  for (auto el : e) {
    std::cout << el << ", ";
  }
  std::cout << std::endl;

  // As it is not used the m-1-th element of e and of mat_trid(1, m-1) are set to 0
  // to make the matrix check happy.
  e[m - 1] = 0.f;
  mat_trid(GlobalTileIndex{0, (m - 1) / mb}).get()({1, (m - 1) % mb}) = 0.f;

  auto res = [&d, &e](const GlobalElementIndex& index) {
    if (index.row() == 0)
      return d[index.col()];
    return e[index.col()];
  };
  // TODO: Find another method as tridiagonalization is not unique.
  // CHECK_MATRIX_NEAR(res, mat_trid, mb * m * TypeUtilities<T>::error, m * TypeUtilities<T>::error);
}

TYPED_TEST(EigensolverBandToTridiagTest, CorrectnessLocal) {
  SizeType m, mb, b;
  const blas::Uplo uplo = blas::Uplo::Lower;

  for (auto sz : sizes) {
    std::tie(m, mb, b) = sz;
    std::cout << "\n----------------------------\n" << std::endl;
    std::cout << m << ", " << mb << ":" << b << std::endl;

    testBandToTridiag<TypeParam>(uplo, b, m, mb);
    std::cout << "\n----------------------------\n" << std::endl;
  }
}
