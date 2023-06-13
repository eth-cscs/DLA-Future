//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <mpi.h>

#include <pika/runtime.hpp>

#include "dlaf_c/eigensolver/eigensolver.h"
#include "dlaf_c/grid.h"
#include "dlaf_c/init.h"
#include "dlaf_c/utils.h"
#include "test_eigensolver_c_api_wrapper.h"

#include <gtest/gtest.h>

TEST(EigensolverCAPIScaLAPACKTest, CorrectnessDistributed) {}
