//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <mkl_service.h>

int main() {
  [[maybe_unused]] auto mkl_num_threads = mkl_set_num_threads_local(1);
}
