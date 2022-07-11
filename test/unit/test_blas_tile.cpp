//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/blas/tile.h"

#include "gtest/gtest.h"

#include "test_blas_tile/test_gemm.h"
#include "test_blas_tile/test_hemm.h"
#include "test_blas_tile/test_her2k.h"
#include "test_blas_tile/test_herk.h"
#include "test_blas_tile/test_trmm.h"
#include "test_blas_tile/test_trsm.h"

using namespace dlaf;
using namespace dlaf::test;
using namespace testing;

const std::vector<blas::Diag> blas_diags({blas::Diag::Unit, blas::Diag::NonUnit});
const std::vector<blas::Op> blas_ops({blas::Op::NoTrans, blas::Op::Trans, blas::Op::ConjTrans});
const std::vector<blas::Side> blas_sides({blas::Side::Left, blas::Side::Right});
const std::vector<blas::Uplo> blas_uplos({blas::Uplo::Lower, blas::Uplo::Upper});

template <class T, Device D>
class TileOperationsTest : public ::testing::Test {};

template <class T>
using TileOperationsTestMC = TileOperationsTest<T, Device::CPU>;

TYPED_TEST_SUITE(TileOperationsTestMC, MatrixElementTypes);

#ifdef DLAF_WITH_GPU
template <class T>
using TileOperationsTestGPU = TileOperationsTest<T, Device::GPU>;

TYPED_TEST_SUITE(TileOperationsTestGPU, MatrixElementTypes);
#endif

// Tuple elements:  m, n, k, extra_lda, extra_ldb, extra_ldc
std::vector<std::tuple<SizeType, SizeType, SizeType, SizeType, SizeType, SizeType>> gemm_sizes = {
    {0, 0, 0, 0, 0, 0},                                               // all 0 sizes
    {7, 0, 0, 3, 1, 0},  {0, 5, 0, 0, 0, 1},    {0, 0, 11, 1, 1, 2},  // two 0 sizes
    {0, 5, 13, 1, 0, 1}, {7, 0, 4, 1, 2, 0},    {3, 11, 0, 0, 1, 0},  // one 0 size
    {1, 1, 1, 0, 3, 0},  {1, 12, 1, 1, 0, 7},   {17, 12, 16, 1, 3, 0}, {11, 23, 8, 0, 3, 4},
    {6, 9, 12, 1, 1, 1}, {32, 32, 32, 0, 0, 0}, {32, 32, 32, 4, 5, 7}, {128, 128, 128, 0, 0, 0},
};

TYPED_TEST(TileOperationsTestMC, Gemm) {
  using Type = TypeParam;

  for (const auto op_a : blas_ops) {
    for (const auto op_b : blas_ops) {
      for (const auto& [m, n, k, extra_lda, extra_ldb, extra_ldc] : gemm_sizes) {
        // Test a and b const Tiles.
        testGemm<Device::CPU, Type>(op_a, op_b, m, n, k, extra_lda, extra_ldb, extra_ldc);

        // Test a and b non const Tiles.
        testGemm<Device::CPU, Type, Type>(op_a, op_b, m, n, k, extra_lda, extra_ldb, extra_ldc);
      }
    }
  }
}

#ifdef DLAF_WITH_GPU
TYPED_TEST(TileOperationsTestGPU, Gemm) {
  using Type = TypeParam;

  for (const auto op_a : blas_ops) {
    for (const auto op_b : blas_ops) {
      for (const auto& [m, n, k, extra_lda, extra_ldb, extra_ldc] : gemm_sizes) {
        // Test a and b const Tiles.
        testGemm<Device::GPU, Type>(op_a, op_b, m, n, k, extra_lda, extra_ldb, extra_ldc);

        // Test a and b non const Tiles.
        testGemm<Device::GPU, Type, Type>(op_a, op_b, m, n, k, extra_lda, extra_ldb, extra_ldc);
      }
    }
  }
}
#endif

// Tuple elements:  m, n, extra_lda, extra_ldb, extra_ldc
std::vector<std::tuple<SizeType, SizeType, SizeType, SizeType, SizeType>> hemm_sizes = {
    {0, 0, 0, 0, 0},                                       // all 0 sizes
    {7, 0, 3, 1, 0}, {0, 5, 0, 0, 1},   {0, 0, 1, 1, 2},   // two 0 sizes
    {0, 5, 1, 0, 1}, {7, 0, 1, 2, 0},   {3, 11, 0, 1, 0},  // one 0 size
    {1, 1, 0, 3, 0}, {1, 12, 1, 0, 7},  {17, 12, 1, 3, 0}, {11, 23, 0, 3, 4},
    {6, 9, 1, 1, 1}, {32, 32, 0, 0, 0}, {32, 32, 4, 5, 7}, {128, 128, 0, 0, 0},
};

TYPED_TEST(TileOperationsTestMC, Hemm) {
  using Type = TypeParam;

  for (const auto side : blas_sides) {
    for (const auto uplo : blas_uplos) {
      for (const auto& [m, n, extra_lda, extra_ldb, extra_ldc] : hemm_sizes) {
        // Test a and b const Tiles.
        testHemm<Device::CPU, Type>(side, uplo, m, n, extra_lda, extra_ldb, extra_ldc);

        // Test a and b non const Tiles.
        testHemm<Device::CPU, Type, Type>(side, uplo, m, n, extra_lda, extra_ldb, extra_ldc);
      }
    }
  }
}

#ifdef DLAF_WITH_GPU
TYPED_TEST(TileOperationsTestGPU, Hemm) {
  using Type = TypeParam;

  for (const auto side : blas_sides) {
    for (const auto uplo : blas_uplos) {
      for (const auto& [m, n, extra_lda, extra_ldb, extra_ldc] : hemm_sizes) {
        // Test a and b const Tiles.
        testHemm<Device::GPU, Type>(side, uplo, m, n, extra_lda, extra_ldb, extra_ldc);

        // Test a and b non const Tiles.
        testHemm<Device::GPU, Type, Type>(side, uplo, m, n, extra_lda, extra_ldb, extra_ldc);
      }
    }
  }
}
#endif

// Tuple elements:  n, k, extra_lda, extra_ldc
std::vector<std::tuple<SizeType, SizeType, SizeType, SizeType>> herk_her2k_sizes = {
    {0, 0, 0, 0},                 // all 0 sizes
    {0, 5, 1, 0},  {7, 0, 1, 2},  // one 0 size
    {1, 1, 0, 3},  {1, 12, 1, 0},  {17, 12, 1, 3}, {11, 23, 0, 3},
    {9, 12, 1, 1}, {32, 32, 0, 0}, {32, 32, 4, 7}, {128, 128, 0, 0},
};

TYPED_TEST(TileOperationsTestMC, Her2k) {
  using Type = TypeParam;

  auto her2k_blas_ops = blas_ops;
  // [c,z]her2k do not allow op = Trans
  if (std::is_same_v<Type, ComplexType<Type>>)
    her2k_blas_ops = {blas::Op::NoTrans, blas::Op::ConjTrans};

  for (const auto uplo : blas_uplos) {
    for (const auto op : her2k_blas_ops) {
      for (const auto& [n, k, extra_lda, extra_ldc] : herk_her2k_sizes) {
        // Test a const Tile.
        testHer2k<Device::CPU, Type>(uplo, op, n, k, extra_lda, extra_ldc);

        // Test a non const Tile.
        testHer2k<Device::CPU, Type, Type>(uplo, op, n, k, extra_lda, extra_ldc);
      }
    }
  }
}

#ifdef DLAF_WITH_GPU
TYPED_TEST(TileOperationsTestGPU, Her2k) {
  using Type = TypeParam;

  auto her2k_blas_ops = blas_ops;
  // [c,z]her2k do not allow op = Trans
  if (std::is_same_v<Type, ComplexType<Type>>)
    her2k_blas_ops = {blas::Op::NoTrans, blas::Op::ConjTrans};

  for (const auto uplo : blas_uplos) {
    for (const auto op : her2k_blas_ops) {
      for (const auto& [n, k, extra_lda, extra_ldc] : herk_her2k_sizes) {
        // Test a const Tile.
        testHer2k<Device::GPU, Type>(uplo, op, n, k, extra_lda, extra_ldc);

        // Test a non const Tile.
        testHer2k<Device::GPU, Type, Type>(uplo, op, n, k, extra_lda, extra_ldc);
      }
    }
  }
}
#endif

TYPED_TEST(TileOperationsTestMC, Herk) {
  using Type = TypeParam;

  auto herk_blas_ops = blas_ops;
  // [c,z]herk do not allow op = Trans
  if (std::is_same_v<Type, ComplexType<Type>>)
    herk_blas_ops = {blas::Op::NoTrans, blas::Op::ConjTrans};

  for (const auto uplo : blas_uplos) {
    for (const auto op : herk_blas_ops) {
      for (const auto& [n, k, extra_lda, extra_ldc] : herk_her2k_sizes) {
        // Test a const Tile.
        testHerk<Device::CPU, Type>(uplo, op, n, k, extra_lda, extra_ldc);

        // Test a non const Tile.
        testHerk<Device::CPU, Type, Type>(uplo, op, n, k, extra_lda, extra_ldc);
      }
    }
  }
}

#ifdef DLAF_WITH_GPU
TYPED_TEST(TileOperationsTestGPU, Herk) {
  using Type = TypeParam;

  auto herk_blas_ops = blas_ops;
  // [c,z]herk do not allow op = Trans
  if (std::is_same_v<Type, ComplexType<Type>>)
    herk_blas_ops = {blas::Op::NoTrans, blas::Op::ConjTrans};

  for (const auto uplo : blas_uplos) {
    for (const auto op : herk_blas_ops) {
      for (const auto& [n, k, extra_lda, extra_ldc] : herk_her2k_sizes) {
        // Test a const Tile.
        testHerk<Device::GPU, Type>(uplo, op, n, k, extra_lda, extra_ldc);

        // Test a non const Tile.
        testHerk<Device::GPU, Type, Type>(uplo, op, n, k, extra_lda, extra_ldc);
      }
    }
  }
}
#endif

// Tuple elements:  m, n, extra_lda, extra_ldb
std::vector<std::tuple<SizeType, SizeType, SizeType, SizeType>> trmm_trsm_sizes = {
    {0, 0, 0, 0},                 // all 0 sizes
    {0, 5, 1, 0},  {7, 0, 1, 2},  // one 0 size
    {1, 1, 0, 3},  {1, 12, 1, 0},  {17, 12, 1, 3}, {11, 23, 0, 3},
    {9, 12, 1, 1}, {32, 32, 0, 0}, {32, 32, 4, 7},
};

// Tuple elements:  m, n, extra_lda, extra_ldb, extra_ldc
std::vector<std::tuple<SizeType, SizeType, SizeType, SizeType, SizeType>> trmm3_sizes = {
    {0, 0, 0, 0, 0},                    // all 0 sizes
    {0, 5, 1, 0, 3},  {7, 0, 1, 2, 0},  // one 0 size
    {1, 1, 0, 3, 2},  {1, 12, 1, 0, 0},  {17, 12, 1, 3, 2}, {11, 23, 0, 3, 0},
    {9, 12, 1, 1, 1}, {32, 32, 0, 0, 2}, {32, 32, 4, 7, 0}, {128, 128, 0, 0, 0},
};

TYPED_TEST(TileOperationsTestMC, Trmm) {
  using Type = TypeParam;

  for (const auto side : blas_sides) {
    for (const auto uplo : blas_uplos) {
      for (const auto op : blas_ops) {
        for (const auto diag : blas_diags) {
          for (const auto& [m, n, extra_lda, extra_ldb] : trmm_trsm_sizes) {
            // Test a const Tile.
            testTrmm<Device::CPU, Type>(side, uplo, op, diag, m, n, extra_lda, extra_ldb);

            // Test a non const Tile.
            testTrmm<Device::CPU, Type, Type>(side, uplo, op, diag, m, n, extra_lda, extra_ldb);
          }
        }
      }
    }
  }
}

TYPED_TEST(TileOperationsTestMC, Trmm3) {
  using Type = TypeParam;

  for (const auto side : blas_sides) {
    for (const auto uplo : blas_uplos) {
      for (const auto op : blas_ops) {
        for (const auto diag : blas_diags) {
          for (const auto& [m, n, extra_lda, extra_ldb, extra_ldc] : trmm3_sizes) {
            // Test a const Tile.
            testTrmm3<Device::CPU, Type>(side, uplo, op, diag, m, n, extra_lda, extra_ldb, extra_ldc);

            // Test a non const Tile.
            testTrmm3<Device::CPU, Type, Type>(side, uplo, op, diag, m, n, extra_lda, extra_ldb,
                                               extra_ldc);
          }
        }
      }
    }
  }
}
#ifdef DLAF_WITH_GPU
TYPED_TEST(TileOperationsTestGPU, Trmm) {
  using Type = TypeParam;

  for (const auto side : blas_sides) {
    for (const auto uplo : blas_uplos) {
      for (const auto op : blas_ops) {
        for (const auto diag : blas_diags) {
          for (const auto& [m, n, extra_lda, extra_ldb] : trmm_trsm_sizes) {
            // Test a const Tile.
            testTrmm<Device::GPU, Type>(side, uplo, op, diag, m, n, extra_lda, extra_ldb);

            // Test a non const Tile.
            testTrmm<Device::GPU, Type, Type>(side, uplo, op, diag, m, n, extra_lda, extra_ldb);
          }
        }
      }
    }
  }
}

TYPED_TEST(TileOperationsTestGPU, Trmm3) {
  using Type = TypeParam;

  for (const auto side : blas_sides) {
    for (const auto uplo : blas_uplos) {
      for (const auto op : blas_ops) {
        for (const auto diag : blas_diags) {
          for (const auto& [m, n, extra_lda, extra_ldb, extra_ldc] : trmm3_sizes) {
            // Test a const Tile.
            testTrmm3<Device::GPU, Type>(side, uplo, op, diag, m, n, extra_lda, extra_ldb, extra_ldc);

            // Test a non const Tile.
            testTrmm3<Device::GPU, Type, Type>(side, uplo, op, diag, m, n, extra_lda, extra_ldb,
                                               extra_ldc);
          }
        }
      }
    }
  }
}
#endif

TYPED_TEST(TileOperationsTestMC, Trsm) {
  using Type = TypeParam;

  for (const auto side : blas_sides) {
    for (const auto uplo : blas_uplos) {
      for (const auto op : blas_ops) {
        for (const auto diag : blas_diags) {
          for (const auto& [m, n, extra_lda, extra_ldb] : trmm_trsm_sizes) {
            // Test a const Tile.
            testTrsm<Device::CPU, Type>(side, uplo, op, diag, m, n, extra_lda, extra_ldb);

            // Test a non const Tile.
            testTrsm<Device::CPU, Type, Type>(side, uplo, op, diag, m, n, extra_lda, extra_ldb);
          }
        }
      }
    }
  }
}

#ifdef DLAF_WITH_GPU
TYPED_TEST(TileOperationsTestGPU, Trsm) {
  using Type = TypeParam;

  for (const auto side : blas_sides) {
    for (const auto uplo : blas_uplos) {
      for (const auto op : blas_ops) {
        for (const auto diag : blas_diags) {
          for (const auto& [m, n, extra_lda, extra_ldb] : trmm_trsm_sizes) {
            // Test a const Tile.
            testTrsm<Device::GPU, Type>(side, uplo, op, diag, m, n, extra_lda, extra_ldb);

            // Test a non const Tile.
            testTrsm<Device::GPU, Type, Type>(side, uplo, op, diag, m, n, extra_lda, extra_ldb);
          }
        }
      }
    }
  }
}
#endif
