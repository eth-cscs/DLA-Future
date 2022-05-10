//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// @file

namespace dlaf {
namespace internal {

/// Computes the i-th updated eigenvalue of a symmetric rank-1 modification to a diagonal matrix:
///
///         D + rho * zz^T
///
/// with rho > 0, D[i] < D[j] for i < j and the Euclidean norm of z is 1.
///
/// @tparam T
///    a real number : either `float` or `double`
///
/// @param[in] d
///    the orginal eigenvalues assumed to be in ascending order : d[i] < d[j] for i < j
///
/// @param[in] z
///    the rank-1 updating vector
///
/// @param[in] rho
///    a scalar multiplication factor : rho > 0
///
/// @param[in] i
///    the index of the eigenvalue to be computed : 0 <= i < n
///
/// @param[out] delta
///    a vector used to construct the eigenvectors of the form : d[j] - lambda
///
/// @param[out] lambda
///    the i-th updated eigenvalue
///
/// This overload blocks until completion of the algorithm.
void laed4_wrapper(int n, int i, float const* d, float const* z, float* delta, float rho,
                   float* lambda) noexcept;

void laed4_wrapper(int n, int i, double const* d, double const* z, double* delta, double rho,
                   double* lambda) noexcept;

}
}
