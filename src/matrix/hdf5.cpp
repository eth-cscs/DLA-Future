//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#ifdef DLAF_WITH_HDF5
#include "dlaf/matrix/hdf5.h"

#include <H5Cpp.h>

namespace dlaf::matrix::internal {

// clang-format off
template <> const H5::PredType& hdf5_datatype<float>  ::type = H5::PredType::NATIVE_FLOAT;
template <> const H5::PredType& hdf5_datatype<double> ::type = H5::PredType::NATIVE_DOUBLE;
// clang-format on

}
#endif
