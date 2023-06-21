//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <dlaf_c/utils.h>

/// Initialize pika runtime and DLA-Future
///
/// The pika runtime is automatically suspended within this function
///
/// @param argc Number of arguments
/// @param argv Arguments
DLAF_EXTERN_C void dlaf_initialize(int argc, const char** argv);

/// Finalize DLA-Future and pika runtime
DLAF_EXTERN_C void dlaf_finalize();
