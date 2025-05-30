//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <complex>
#include <optional>
#include <set>
#include <tuple>
#include <typeinfo>
#include <utility>
#include <vector>

#include <blas/util.hh>

#include <pika/init.hpp>

#include <dlaf/communication/communicator_grid.h>
#include <dlaf/eigensolver/eigensolver.h>
#include <dlaf/matrix/copy.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf_c_test/c_api_helpers.h>

#include "test_eigensolver_c_api_config.h"
#include "test_eigensolver_c_api_wrapper.h"

#include <gtest/gtest.h>

#include <dlaf_test/comm_grids/grids_6_ranks.h>
#include <dlaf_test/eigensolver/test_eigensolver_correctness.h>
#include <dlaf_test/matrix/util_matrix.h>

using namespace dlaf;
using namespace dlaf::comm;
using namespace dlaf::matrix;
using namespace dlaf::test;
using namespace testing;

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksCAPIEnvironment);

template <typename Type>
class EigensolverTest : public TestWithCommGrids {};

template <class T>
using EigensolverTestCapi = EigensolverTest<T>;

TYPED_TEST_SUITE(EigensolverTestCapi, MatrixElementTypes);

const std::vector<blas::Uplo> blas_uplos({blas::Uplo::Lower});

const std::vector<std::tuple<SizeType, SizeType, SizeType>> sizes = {
    // {m, mb, eigensolver_min_band}
    {0, 2, 100},                                              // m = 0
    {5, 8, 100}, {34, 34, 100},                               // m <= mb
    {4, 3, 100}, {16, 10, 100}, {34, 13, 100}, {32, 5, 100},  // m > mb
    {34, 8, 3},  {32, 6, 3}                                   // m > mb, sub-band
};

std::set<std::optional<SizeType>> num_evals(const SizeType m) {
  return {std::nullopt, 0, m / 2, m};
}

template <class T, API api>
void testEigensolver(int dlaf_context, const blas::Uplo uplo, const SizeType m, const SizeType mb,
                     CommunicatorGrid& grid, std::optional<SizeType> eigenvalues_index_end) {
  // std::nullopt calls the API without specifying the number of eigenvalues to compute
  // The final check needs to happen on all m eigenvalues/eigenvectors
  const SizeType eval_idx_end = eigenvalues_index_end.value_or(m);

  // In normal use the runtime is resumed by the C API call
  // The pika runtime is suspended by dlaf_initialize
  // Here we need to resume it manually to build the matrices with DLA-Future
  pika::resume();

  const TileElementSize block_size(mb, mb);

  Matrix<const T, Device::CPU> reference = [&]() {
    auto reference = [&]() -> auto {
      return Matrix<T, Device::CPU>(GlobalElementSize(m, m), block_size, grid);
    }();
    matrix::util::set_random_hermitian(reference);
    return reference;
  }();

  Matrix<T, Device::CPU> mat_a_h(reference.distribution());
  copy(reference, mat_a_h);
  mat_a_h.waitLocalTiles();

  EigensolverResult<T, Device::CPU> ret = [&]() {
    const SizeType size = mat_a_h.size().rows();
    Matrix<BaseType<T>, Device::CPU> eigenvalues(LocalElementSize(size, 1),
                                                 TileElementSize(mat_a_h.blockSize().rows(), 1));
    Matrix<T, Device::CPU> eigenvectors(GlobalElementSize(size, size), mat_a_h.blockSize(), grid);

    eigenvalues.waitLocalTiles();
    eigenvectors.waitLocalTiles();

    char dlaf_uplo = blas::to_char(uplo);

    // Get top left local tiles
    auto [local_a_ptr, lld_a] = top_left_tile(mat_a_h);
    auto [local_eigenvectors_ptr, lld_eigenvectors] = top_left_tile(eigenvectors);
    auto [eigenvalues_ptr, lld_eigenvalues] = top_left_tile(eigenvalues);

    // Suspend pika to ensure it is resumed by the C API
    pika::suspend();

    if constexpr (api == API::dlaf) {
      DLAF_descriptor dlaf_desc_a = {(int) m, (int) m, (int) mb, (int) mb, 0, 0, 0, 0, lld_a};
      DLAF_descriptor dlaf_desc_eigenvectors = {(int) m, (int) m, (int) mb, (int) mb,        0,
                                                0,       0,       0,        lld_eigenvectors};

      int err = -1;
      if constexpr (std::is_same_v<T, double>) {
        if (eigenvalues_index_end.has_value()) {
          err = C_dlaf_symmetric_eigensolver_partial_spectrum_d(dlaf_context, dlaf_uplo, local_a_ptr,
                                                                dlaf_desc_a, eigenvalues_ptr,
                                                                local_eigenvectors_ptr,
                                                                dlaf_desc_eigenvectors, 0, eval_idx_end);
        }
        else {
          err = C_dlaf_symmetric_eigensolver_d(dlaf_context, dlaf_uplo, local_a_ptr, dlaf_desc_a,
                                               eigenvalues_ptr, local_eigenvectors_ptr,
                                               dlaf_desc_eigenvectors);
        }
      }
      else if constexpr (std::is_same_v<T, float>) {
        if (eigenvalues_index_end.has_value()) {
          err = C_dlaf_symmetric_eigensolver_partial_spectrum_s(dlaf_context, dlaf_uplo, local_a_ptr,
                                                                dlaf_desc_a, eigenvalues_ptr,
                                                                local_eigenvectors_ptr,
                                                                dlaf_desc_eigenvectors, 0, eval_idx_end);
        }
        else {
          err = C_dlaf_symmetric_eigensolver_s(dlaf_context, dlaf_uplo, local_a_ptr, dlaf_desc_a,
                                               eigenvalues_ptr, local_eigenvectors_ptr,
                                               dlaf_desc_eigenvectors);
        }
      }
      else if constexpr (std::is_same_v<T, std::complex<double>>) {
        if (eigenvalues_index_end.has_value()) {
          err = C_dlaf_hermitian_eigensolver_partial_spectrum_z(dlaf_context, dlaf_uplo, local_a_ptr,
                                                                dlaf_desc_a, eigenvalues_ptr,
                                                                local_eigenvectors_ptr,
                                                                dlaf_desc_eigenvectors, 0, eval_idx_end);
        }
        else {
          err = C_dlaf_hermitian_eigensolver_z(dlaf_context, dlaf_uplo, local_a_ptr, dlaf_desc_a,
                                               eigenvalues_ptr, local_eigenvectors_ptr,
                                               dlaf_desc_eigenvectors);
        }
      }
      else if constexpr (std::is_same_v<T, std::complex<float>>) {
        if (eigenvalues_index_end.has_value()) {
          err = C_dlaf_hermitian_eigensolver_partial_spectrum_c(dlaf_context, dlaf_uplo, local_a_ptr,
                                                                dlaf_desc_a, eigenvalues_ptr,
                                                                local_eigenvectors_ptr,
                                                                dlaf_desc_eigenvectors, 0, eval_idx_end);
        }
        else {
          err = C_dlaf_hermitian_eigensolver_c(dlaf_context, dlaf_uplo, local_a_ptr, dlaf_desc_a,
                                               eigenvalues_ptr, local_eigenvectors_ptr,
                                               dlaf_desc_eigenvectors);
        }
      }
      else {
        DLAF_ASSERT(false, typeid(T).name());
      }
      DLAF_ASSERT(err == 0, err);
    }
    else if constexpr (api == API::scalapack) {
#ifdef DLAF_WITH_SCALAPACK
      int desc_a[] = {1, dlaf_context, (int) m, (int) m, (int) mb, (int) mb, 0, 0, lld_a};
      int desc_z[] = {1, dlaf_context, (int) m, (int) m, (int) mb, (int) mb, 0, 0, lld_eigenvectors};
      int info = -1;

      // Treat special case when eval_idx_end is 0 for the C API
      // The ScaLAPACK API uses base 1 indexing
      const SizeType eval_idx_end_scalapack = m > 0 && eval_idx_end == 0 ? 1 : eval_idx_end;

      if constexpr (std::is_same_v<T, double>) {
        if (eigenvalues_index_end.has_value())
          C_dlaf_pdsyevd_partial_spectrum(dlaf_uplo, (int) m, local_a_ptr, 1, 1, desc_a, eigenvalues_ptr,
                                          local_eigenvectors_ptr, 1, 1, desc_z, 1,
                                          eval_idx_end_scalapack, &info);
        else
          C_dlaf_pdsyevd(dlaf_uplo, (int) m, local_a_ptr, 1, 1, desc_a, eigenvalues_ptr,
                         local_eigenvectors_ptr, 1, 1, desc_z, &info);
      }
      else if constexpr (std::is_same_v<T, float>) {
        if (eigenvalues_index_end.has_value())
          C_dlaf_pssyevd_partial_spectrum(dlaf_uplo, (int) m, local_a_ptr, 1, 1, desc_a, eigenvalues_ptr,
                                          local_eigenvectors_ptr, 1, 1, desc_z, 1,
                                          eval_idx_end_scalapack, &info);
        else
          C_dlaf_pssyevd(dlaf_uplo, (int) m, local_a_ptr, 1, 1, desc_a, eigenvalues_ptr,
                         local_eigenvectors_ptr, 1, 1, desc_z, &info);
      }
      else if constexpr (std::is_same_v<T, std::complex<double>>) {
        if (eigenvalues_index_end.has_value())
          C_dlaf_pzheevd_partial_spectrum(dlaf_uplo, (int) m, local_a_ptr, 1, 1, desc_a, eigenvalues_ptr,
                                          local_eigenvectors_ptr, 1, 1, desc_z, 1,
                                          eval_idx_end_scalapack, &info);
        else
          C_dlaf_pzheevd(dlaf_uplo, (int) m, local_a_ptr, 1, 1, desc_a, eigenvalues_ptr,
                         local_eigenvectors_ptr, 1, 1, desc_z, &info);
      }
      else if constexpr (std::is_same_v<T, std::complex<float>>) {
        if (eigenvalues_index_end.has_value())
          C_dlaf_pcheevd_partial_spectrum(dlaf_uplo, (int) m, local_a_ptr, 1, 1, desc_a, eigenvalues_ptr,
                                          local_eigenvectors_ptr, 1, 1, desc_z, 1,
                                          eval_idx_end_scalapack, &info);
        else
          C_dlaf_pcheevd(dlaf_uplo, (int) m, local_a_ptr, 1, 1, desc_a, eigenvalues_ptr,
                         local_eigenvectors_ptr, 1, 1, desc_z, &info);
      }
      else {
        DLAF_ASSERT(false, typeid(T).name());
      }
      DLAF_ASSERT(info == 0, info);
#else
      static_assert(api != API::scalapack, "DLA-Future compiled without ScaLAPACK support.");
#endif
    }

    return EigensolverResult<T, Device::CPU>{std::move(eigenvalues), std::move(eigenvectors)};
  }();

  // Resume pika runtime suspended by C API for correctness checks
  pika::resume();

  if (!mat_a_h.size().isEmpty() && eval_idx_end != 0) {
    testEigensolverCorrectness(uplo, reference, ret.eigenvalues, ret.eigenvectors, 0l, eval_idx_end,
                               grid);
  }

  // Suspend pika to make sure dlaf_finalize resumes it
  pika::suspend();
}

TYPED_TEST(EigensolverTestCapi, CorrectnessDistributedDLAF) {
  for (comm::CommunicatorGrid& grid : this->commGrids()) {
    auto dlaf_context =
        c_api_test_initialize<API::dlaf>(pika_argc, pika_argv, dlaf_argc, dlaf_argv, grid);

    for (auto uplo : blas_uplos) {
      for (auto [m, mb, b_min] : sizes) {
        auto numevals = num_evals(m);
        for (auto nevals : numevals) {
          testEigensolver<TypeParam, API::dlaf>(dlaf_context, uplo, m, mb, grid, nevals);
        }
      }
    }

    c_api_test_finalize<API::dlaf>(dlaf_context);
  }
}

#ifdef DLAF_WITH_SCALAPACK
TYPED_TEST(EigensolverTestCapi, CorrectnessDistributedScalapack) {
  for (comm::CommunicatorGrid& grid : this->commGrids()) {
    auto dlaf_context =
        c_api_test_initialize<API::scalapack>(pika_argc, pika_argv, dlaf_argc, dlaf_argv, grid);

    for (auto uplo : blas_uplos) {
      for (auto [m, mb, b_min] : sizes) {
        auto numevals = num_evals(m);
        for (auto nevals : numevals) {
          testEigensolver<TypeParam, API::scalapack>(dlaf_context, uplo, m, mb, grid, nevals);
        }
      }
    }

    c_api_test_finalize<API::scalapack>(dlaf_context);
  }
}
#endif
