//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// @file

#include <dlaf_c/utils.h>

/// Initialize pika runtime and DLA-Future
///
/// @remark If DLA-Future has already been initialized, this function does nothing
///
/// @post The pika runtime is automatically suspended when this function returns
///
/// @param argc_pika Number of arguments for pika
/// @param argv_pika Arguments for pika
/// @param argc_dlaf Number of arguments for DLA-Future
/// @param argv_dlaf Arguments for DLA-Future
DLAF_EXTERN_C void dlaf_initialize(int argc_pika, const char** argv_pika, int argc_dlaf,
                                   const char** argv_dlaf) DLAF_NOEXCEPT;

/// Finalize DLA-Future and pika runtime
///
/// @post The pika runtime is finalized and stopped when this function returns
///
/// @remark If DLA-Future has already been finalized, this function does nothing
DLAF_EXTERN_C void dlaf_finalize() DLAF_NOEXCEPT;
