//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <functional>
#include <tuple>
#include <typeinfo>
#include <utility>
#include <vector>

#include <pika/init.hpp>

#include <dlaf/communication/communicator_grid.h>
#include <dlaf/eigensolver/gen_eigensolver.h>
#include <dlaf/matrix/copy.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf_c_test/c_api_helpers.h>

#include "../factorization/test_cholesky_c_api_wrapper.h"
#include "test_gen_eigensolver_c_api_config.h"
#include "test_gen_eigensolver_c_api_wrapper.h"

#include <gtest/gtest.h>

#include <dlaf_test/comm_grids/grids_6_ranks.h>
#include <dlaf_test/eigensolver/test_gen_eigensolver_correctness.h>
#include <dlaf_test/matrix/util_matrix.h>

using namespace dlaf;
using namespace dlaf::comm;
using namespace dlaf::matrix;
using namespace dlaf::test;
using namespace testing;

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksCAPIEnvironment);

template <typename Type>
class GenEigensolverTest : public TestWithCommGrids {};

template <class T>
using GenEigensolverTestMC = GenEigensolverTest<T>;

TYPED_TEST_SUITE(GenEigensolverTestMC, MatrixElementTypes);

#ifdef DLAF_WITH_GPU
template <class T>
using GenEigensolverTestGPU = GenEigensolverTest<T>;

TYPED_TEST_SUITE(GenEigensolverTestGPU, MatrixElementTypes);
#endif

using dlaf::eigensolver::internal::Factorization;

const std::vector<blas::Uplo> blas_uplos({blas::Uplo::Lower});

const std::vector<std::tuple<SizeType, SizeType, SizeType>> sizes = {
    // {m, mb, eigensolver_min_band}
    {0, 2, 100},                                              // m = 0
    {5, 8, 100}, {34, 34, 100},                               // m <= mb
    {4, 3, 100}, {16, 10, 100}, {34, 13, 100}, {32, 5, 100},  // m > mb
    {34, 8, 3},  {32, 6, 3}                                   // m > mb, sub-band
};

template <class T, Backend B, Device D, API api, Factorization factorization>
void testGenEigensolver(const blas::Uplo uplo, const SizeType m, const SizeType mb,
                        CommunicatorGrid& grid) {
  auto dlaf_context = c_api_test_inititialize<api>(pika_argc, pika_argv, dlaf_argc, dlaf_argv, grid);

  // In normal use the runtime is resumed by the C API call
  // The pika runtime is suspended by dlaf_initialize
  // Here we need to resume it manually to build the matrices with DLA-Future
  pika::resume();

  const TileElementSize block_size(mb, mb);

  auto create_reference = [&]() {
    return Matrix<T, Device::CPU>(GlobalElementSize(m, m), block_size, grid);
  };

  Matrix<const T, Device::CPU> reference_a = [&]() {
    auto reference = create_reference();
    matrix::util::set_random_hermitian(reference);
    return reference;
  }();

  Matrix<const T, Device::CPU> reference_b = [&]() {
    auto reference = create_reference();
    matrix::util::set_random_hermitian_positive_definite(reference);
    return reference;
  }();

  Matrix<T, Device::CPU> mat_a_h(reference_a.distribution());
  copy(reference_a, mat_a_h);
  mat_a_h.waitLocalTiles();
  Matrix<T, Device::CPU> mat_b_h(reference_b.distribution());
  copy(reference_b, mat_b_h);
  mat_b_h.waitLocalTiles();

  EigensolverResult<T, D> ret = [&]() {
    MatrixMirror<T, D, Device::CPU> mat_a(mat_a_h);
    MatrixMirror<T, D, Device::CPU> mat_b(mat_b_h);

    const SizeType size = mat_a_h.size().rows();
    Matrix<BaseType<T>, D> eigenvalues(LocalElementSize(size, 1),
                                       TileElementSize(mat_a_h.blockSize().rows(), 1));
    Matrix<T, D> eigenvectors(GlobalElementSize(size, size), mat_a_h.blockSize(), grid);

    eigenvalues.waitLocalTiles();
    eigenvectors.waitLocalTiles();

    char dlaf_uplo = blas::uplo2char(uplo);

    // Get top left local tiles
    auto [local_a_ptr, lld_a] = top_left_tile(mat_a_h);
    auto [local_b_ptr, lld_b] = top_left_tile(mat_b_h);
    auto [local_eigenvectors_ptr, lld_eigenvectors] = top_left_tile(eigenvectors);
    auto [eigenvalues_ptr, lld_eigenvalues] = top_left_tile(eigenvalues);

    // Suspend pika to ensure it is resumed by the C API
    pika::suspend();

    if constexpr (api == API::dlaf) {
      DLAF_descriptor dlaf_desc_a = {(int) m, (int) m, (int) mb, (int) mb, 0, 0, 0, 0, lld_a};
      DLAF_descriptor dlaf_desc_b = {(int) m, (int) m, (int) mb, (int) mb, 0, 0, 0, 0, lld_b};
      DLAF_descriptor dlaf_desc_eigenvectors = {(int) m, (int) m, (int) mb, (int) mb,        0,
                                                0,       0,       0,        lld_eigenvectors};

      int err = -1;
      if constexpr (std::is_same_v<T, double>) {
        if constexpr (factorization == Factorization::do_factorization) {
          err = C_dlaf_symmetric_generalized_eigensolver_d(dlaf_context, dlaf_uplo, local_a_ptr,
                                                           dlaf_desc_a, local_b_ptr, dlaf_desc_b,
                                                           eigenvalues_ptr, local_eigenvectors_ptr,
                                                           dlaf_desc_eigenvectors);
        }
        else {
          err = C_dlaf_cholesky_factorization_d(dlaf_context, dlaf_uplo, local_b_ptr, dlaf_desc_b);
          DLAF_ASSERT(err == 0, err);

          err = C_dlaf_symmetric_generalized_eigensolver_factorized_d(
              dlaf_context, dlaf_uplo, local_a_ptr, dlaf_desc_a, local_b_ptr, dlaf_desc_b,
              eigenvalues_ptr, local_eigenvectors_ptr, dlaf_desc_eigenvectors);
        }
      }
      else if constexpr (std::is_same_v<T, float>) {
        if constexpr (factorization == Factorization::do_factorization) {
          err = C_dlaf_symmetric_generalized_eigensolver_s(dlaf_context, dlaf_uplo, local_a_ptr,
                                                           dlaf_desc_a, local_b_ptr, dlaf_desc_b,
                                                           eigenvalues_ptr, local_eigenvectors_ptr,
                                                           dlaf_desc_eigenvectors);
        }
        else {
          err = C_dlaf_cholesky_factorization_s(dlaf_context, dlaf_uplo, local_b_ptr, dlaf_desc_b);
          DLAF_ASSERT(err == 0, err);

          err = C_dlaf_symmetric_generalized_eigensolver_factorized_s(
              dlaf_context, dlaf_uplo, local_a_ptr, dlaf_desc_a, local_b_ptr, dlaf_desc_b,
              eigenvalues_ptr, local_eigenvectors_ptr, dlaf_desc_eigenvectors);
        }
      }
      else if constexpr (std::is_same_v<T, std::complex<double>>) {
        if constexpr (factorization == Factorization::do_factorization) {
          err = C_dlaf_hermitian_generalized_eigensolver_z(dlaf_context, dlaf_uplo, local_a_ptr,
                                                           dlaf_desc_a, local_b_ptr, dlaf_desc_b,
                                                           eigenvalues_ptr, local_eigenvectors_ptr,
                                                           dlaf_desc_eigenvectors);
        }
        else {
          err = C_dlaf_cholesky_factorization_z(dlaf_context, dlaf_uplo, local_b_ptr, dlaf_desc_b);
          DLAF_ASSERT(err == 0, err);

          err = C_dlaf_hermitian_generalized_eigensolver_factorized_z(
              dlaf_context, dlaf_uplo, local_a_ptr, dlaf_desc_a, local_b_ptr, dlaf_desc_b,
              eigenvalues_ptr, local_eigenvectors_ptr, dlaf_desc_eigenvectors);
        }
      }
      else if constexpr (std::is_same_v<T, std::complex<float>>) {
        if constexpr (factorization == Factorization::do_factorization) {
          err = C_dlaf_hermitian_generalized_eigensolver_c(dlaf_context, dlaf_uplo, local_a_ptr,
                                                           dlaf_desc_a, local_b_ptr, dlaf_desc_b,
                                                           eigenvalues_ptr, local_eigenvectors_ptr,
                                                           dlaf_desc_eigenvectors);
        }
        else {
          err = C_dlaf_cholesky_factorization_c(dlaf_context, dlaf_uplo, local_b_ptr, dlaf_desc_b);
          DLAF_ASSERT(err == 0, err);

          err = C_dlaf_hermitian_generalized_eigensolver_factorized_c(
              dlaf_context, dlaf_uplo, local_a_ptr, dlaf_desc_a, local_b_ptr, dlaf_desc_b,
              eigenvalues_ptr, local_eigenvectors_ptr, dlaf_desc_eigenvectors);
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
      int desc_b[] = {1, dlaf_context, (int) m, (int) m, (int) mb, (int) mb, 0, 0, lld_b};
      int desc_z[] = {1, dlaf_context, (int) m, (int) m, (int) mb, (int) mb, 0, 0, lld_eigenvectors};
      int info = -1;
      if constexpr (std::is_same_v<T, double>) {
        if constexpr (factorization == Factorization::do_factorization) {
          C_dlaf_pdsygvd(dlaf_uplo, (int) m, local_a_ptr, 1, 1, desc_a, local_b_ptr, 1, 1, desc_b,
                         eigenvalues_ptr, local_eigenvectors_ptr, 1, 1, desc_z, &info);
        }
        else {
          C_dlaf_pdpotrf(dlaf_uplo, (int) m, local_b_ptr, 1, 1, desc_b, &info);
          DLAF_ASSERT(info == 0, info);
          info = -1;

          C_dlaf_pdsygvd_factorized(dlaf_uplo, (int) m, local_a_ptr, 1, 1, desc_a, local_b_ptr, 1, 1,
                                    desc_b, eigenvalues_ptr, local_eigenvectors_ptr, 1, 1, desc_z,
                                    &info);
        }
      }
      else if constexpr (std::is_same_v<T, float>) {
        if constexpr (factorization == Factorization::do_factorization) {
          C_dlaf_pssygvd(dlaf_uplo, (int) m, local_a_ptr, 1, 1, desc_a, local_b_ptr, 1, 1, desc_b,
                         eigenvalues_ptr, local_eigenvectors_ptr, 1, 1, desc_z, &info);
        }
        else {
          C_dlaf_pspotrf(dlaf_uplo, (int) m, local_b_ptr, 1, 1, desc_b, &info);
          DLAF_ASSERT(info == 0, info);
          info = -1;

          C_dlaf_pssygvd_factorized(dlaf_uplo, (int) m, local_a_ptr, 1, 1, desc_a, local_b_ptr, 1, 1,
                                    desc_b, eigenvalues_ptr, local_eigenvectors_ptr, 1, 1, desc_z,
                                    &info);
        }
      }
      else if constexpr (std::is_same_v<T, std::complex<double>>) {
        if constexpr (factorization == Factorization::do_factorization) {
          C_dlaf_pzhegvd(dlaf_uplo, (int) m, local_a_ptr, 1, 1, desc_a, local_b_ptr, 1, 1, desc_b,
                         eigenvalues_ptr, local_eigenvectors_ptr, 1, 1, desc_z, &info);
        }
        else {
          C_dlaf_pzpotrf(dlaf_uplo, (int) m, local_b_ptr, 1, 1, desc_b, &info);
          DLAF_ASSERT(info == 0, info);
          info = -1;

          C_dlaf_pzhegvd_factorized(dlaf_uplo, (int) m, local_a_ptr, 1, 1, desc_a, local_b_ptr, 1, 1,
                                    desc_b, eigenvalues_ptr, local_eigenvectors_ptr, 1, 1, desc_z,
                                    &info);
        }
      }
      else if constexpr (std::is_same_v<T, std::complex<float>>) {
        if constexpr (factorization == Factorization::do_factorization) {
          C_dlaf_pchegvd(dlaf_uplo, (int) m, local_a_ptr, 1, 1, desc_a, local_b_ptr, 1, 1, desc_b,
                         eigenvalues_ptr, local_eigenvectors_ptr, 1, 1, desc_z, &info);
        }
        else {
          C_dlaf_pcpotrf(dlaf_uplo, (int) m, local_b_ptr, 1, 1, desc_b, &info);
          DLAF_ASSERT(info == 0, info);
          info = -1;

          C_dlaf_pchegvd_factorized(dlaf_uplo, (int) m, local_a_ptr, 1, 1, desc_a, local_b_ptr, 1, 1,
                                    desc_b, eigenvalues_ptr, local_eigenvectors_ptr, 1, 1, desc_z,
                                    &info);
        }
      }
      else {
        DLAF_ASSERT(false, typeid(T).name());
      }
      DLAF_ASSERT(info == 0, info);
#else
      static_assert(api != API::scalapack, "DLA-Future compiled without ScaLAPACK support.");
#endif
    }

    return EigensolverResult<T, D>{std::move(eigenvalues), std::move(eigenvectors)};
  }();

  // Resume pika runtime suspended by C API for correctness checks
  pika::resume();

  if (!mat_a_h.size().isEmpty())
    testGenEigensolverCorrectness(uplo, reference_a, reference_b, ret, grid);

  // Suspend pika to make sure dlaf_finalize resumes it
  pika::suspend();

  c_api_test_finalize<api>(dlaf_context);
}

TYPED_TEST(GenEigensolverTestMC, CorrectnessDistributedDLAF) {
  for (comm::CommunicatorGrid& grid : this->commGrids()) {
    for (auto uplo : blas_uplos) {
      for (auto [m, mb, b_min] : sizes) {
        testGenEigensolver<TypeParam, Backend::MC, Device::CPU, API::dlaf,
                           Factorization::do_factorization>(uplo, m, mb, grid);
        testGenEigensolver<TypeParam, Backend::MC, Device::CPU, API::dlaf,
                           Factorization::already_factorized>(uplo, m, mb, grid);
      }
    }
  }
}

#ifdef DLAF_WITH_GPU
TYPED_TEST(GenEigensolverTestGPU, CorrectnessDistributedDLAF) {
  for (comm::CommunicatorGrid& grid : this->commGrids()) {
    for (auto uplo : blas_uplos) {
      for (auto [m, mb, b_min] : sizes) {
        testGenEigensolver<TypeParam, Backend::GPU, Device::GPU, API::dlaf,
                           Factorization::do_factorization>(uplo, m, mb, grid);
        testGenEigensolver<TypeParam, Backend::GPU, Device::GPU, API::dlaf,
                           Factorization::already_factorized>(uplo, m, mb, grid);
      }
    }
  }
}
#endif

#ifdef DLAF_WITH_SCALAPACK

TYPED_TEST(GenEigensolverTestMC, CorrectnessDistributedScaLAPACK) {
  for (comm::CommunicatorGrid& grid : this->commGrids()) {
    for (auto uplo : blas_uplos) {
      for (auto [m, mb, b_min] : sizes) {
        testGenEigensolver<TypeParam, Backend::MC, Device::CPU, API::scalapack,
                           Factorization::do_factorization>(uplo, m, mb, grid);
        testGenEigensolver<TypeParam, Backend::MC, Device::CPU, API::scalapack,
                           Factorization::already_factorized>(uplo, m, mb, grid);
      }
    }
  }
}

#ifdef DLAF_WITH_GPU
TYPED_TEST(GenEigensolverTestGPU, CorrectnessDistributedScaLAPACK) {
  for (comm::CommunicatorGrid& grid : this->commGrids()) {
    for (auto uplo : blas_uplos) {
      for (auto [m, mb, b_min] : sizes) {
        testGenEigensolver<TypeParam, Backend::GPU, Device::GPU, API::scalapack,
                           Factorization::do_factorization>(uplo, m, mb, grid);
        testGenEigensolver<TypeParam, Backend::GPU, Device::GPU, API::scalapack,
                           Factorization::already_factorized>(uplo, m, mb, grid);
      }
    }
  }
}
#endif

#endif
