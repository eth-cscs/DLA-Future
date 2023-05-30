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

namespace dlaf::interface::utils {

extern "C" void dlafuture_init(int argc, const char** argv);
extern "C" void dlafuture_finalize();

void dlaf_check(char uplo, int* desc, int& info);

}
