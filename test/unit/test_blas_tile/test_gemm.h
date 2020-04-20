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

#include <exception>
#include <sstream>
#include "gtest/gtest.h"
#include "dlaf/blas_tile.h"
#include "dlaf/memory/memory_view.h"
#include "dlaf/tile.h"
#include "dlaf/util_blas.h"
#include "dlaf_test/matrix/util_tile.h"
#include "dlaf_test/matrix/util_tile_blas.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace dlaf_test;
using namespace testing;

using dlaf::util::size_t::mul;

template <class T, class CT = const T>
void testGemm(blas::Op op_a, blas::Op op_b, SizeType m, SizeType n, SizeType k, SizeType extra_lda,
              SizeType extra_ldb, SizeType extra_ldc) {
  TileElementSize size_a(m, k);
  if (op_a != blas::Op::NoTrans)
    size_a.transpose();
  TileElementSize size_b(k, n);
  if (op_b != blas::Op::NoTrans)
    size_b.transpose();
  TileElementSize size_c(m, n);

  SizeType lda = std::max<SizeType>(1, size_a.rows()) + extra_lda;
  SizeType ldb = std::max<SizeType>(1, size_b.rows()) + extra_ldb;
  SizeType ldc = std::max<SizeType>(1, size_c.rows()) + extra_ldc;

  std::stringstream s;
  s << "GEMM: " << op_a << ", " << op_a;
  s << ", m = " << m << ", n = " << n << ", k = " << k;
  s << ", lda = " << lda << ", ldb = " << ldb << ", ldc = " << ldc;
  SCOPED_TRACE(s.str());

  memory::MemoryView<T, Device::CPU> mem_a(mul(lda, size_a.cols()));
  memory::MemoryView<T, Device::CPU> mem_b(mul(ldb, size_b.cols()));
  memory::MemoryView<T, Device::CPU> mem_c(mul(ldc, size_c.cols()));

  // Create tiles.
  Tile<T, Device::CPU> a0(size_a, std::move(mem_a), lda);
  Tile<T, Device::CPU> b0(size_b, std::move(mem_b), ldb);
  Tile<T, Device::CPU> c(size_c, std::move(mem_c), ldc);

  // Note: The tile elements are chosen such that:
  // - op_a(a)_ik = .9 * (i+1) / (k+.5) * exp(I*(2*i-k)),
  // - op_b(b)_kj = .8 * (k+.5) / (j+2) * exp(I*(k-j)),
  // - c_ij = 1.2 * i / (j+1) * exp(I*(-i+j)),
  // where I = 0 for real types or I is the complex unit for complex types.
  // Therefore the result should be:
  // res_ij = beta * c_ij + Sum_k(alpha * op_a(a)_ik * op_b(b)_kj)
  //        = beta * c_ij + gamma * (i+1) / (j+2) * exp(I*(2*i+j)),
  // where gamma = .72 * k * alpha.
  auto el_op_a = [](const TileElementIndex& index) {
    double i = index.row();
    double k = index.col();
    return TypeUtilities<T>::polar(.9 * (i + 1) / (k + .5), 2 * i - k);
  };
  auto el_op_b = [](const TileElementIndex& index) {
    double k = index.row();
    double j = index.col();
    return TypeUtilities<T>::polar(.8 * (k + .5) / (j + 2), k + j);
  };
  auto el_c = [](const TileElementIndex& index) {
    double i = index.row();
    double j = index.col();
    return TypeUtilities<T>::polar(1.2 * i / (j + 1), -i + j);
  };

  T alpha = TypeUtilities<T>::element(-1.2, .7);
  T beta = TypeUtilities<T>::element(1.1, .4);

  T gamma = TypeUtilities<T>::element(.72 * k, 0) * alpha;
  auto res_c = [beta, el_c, gamma](const TileElementIndex& index) {
    double i = index.row();
    double j = index.col();
    return beta * el_c(index) + gamma * TypeUtilities<T>::polar((i + 1) / (j + 2), 2 * i + j);
  };

  // Set tile elements.
  set(a0, el_op_a, op_a);
  set(b0, el_op_b, op_b);
  set(c, el_c);

  // Read-only tiles become constant if CT is const T.
  Tile<CT, Device::CPU> a(std::move(a0));
  Tile<CT, Device::CPU> b(std::move(b0));

  tile::gemm(op_a, op_b, alpha, a, b, beta, c);

  // Check result against analytical result.
  CHECK_TILE_NEAR(res_c, c, 2 * (k + 1) * TypeUtilities<T>::error,
                  2 * (k + 1) * TypeUtilities<T>::error);
}

template <class T, class CT = const T>
void testGemmExceptions(blas::Op op_a, blas::Op op_b, const TileElementSize& size_op_a,
                        const TileElementSize& size_op_b, const TileElementSize& size_c,
                        SizeType extra_lda, SizeType extra_ldb, SizeType extra_ldc) {
  TileElementSize size_a = size_op_a;
  if (op_a != blas::Op::NoTrans)
    size_a.transpose();
  TileElementSize size_b = size_op_b;
  if (op_b != blas::Op::NoTrans)
    size_b.transpose();

  SizeType lda = std::max<SizeType>(1, size_a.rows()) + extra_lda;
  SizeType ldb = std::max<SizeType>(1, size_b.rows()) + extra_ldb;
  SizeType ldc = std::max<SizeType>(1, size_c.rows()) + extra_ldc;

  std::stringstream s;
  s << "GEMM Exceptions: " << op_a << ", " << op_a;
  s << ", size_a = " << size_a << ", size_b = " << size_b << ", size_c = " << size_c;
  s << ", lda = " << lda << ", ldb = " << ldb << ", ldc = " << ldc;
  SCOPED_TRACE(s.str());

  memory::MemoryView<T, Device::CPU> mem_a(mul(lda, size_a.cols()));
  memory::MemoryView<T, Device::CPU> mem_b(mul(ldb, size_b.cols()));
  memory::MemoryView<T, Device::CPU> mem_c(mul(ldc, size_c.cols()));

  // Create tiles.
  Tile<CT, Device::CPU> a(size_a, std::move(mem_a), lda);
  Tile<CT, Device::CPU> b(size_b, std::move(mem_b), ldb);
  Tile<T, Device::CPU> c(size_c, std::move(mem_c), ldc);

  T alpha = TypeUtilities<T>::element(-1.2, .7);
  T beta = TypeUtilities<T>::element(1.1, .4);

  EXPECT_THROW(tile::gemm(op_a, op_b, alpha, a, b, beta, c), std::invalid_argument);
}
