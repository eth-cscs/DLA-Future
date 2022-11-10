//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <gtest/gtest.h>

#include "dlaf/lapack/tile.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/sender/policy.h"
#include "dlaf/sender/with_temporary_tile.h"
#include "dlaf/types.h"
#include "dlaf_test/matrix/util_tile.h"

using namespace dlaf;
namespace ex = pika::execution::experimental;
namespace tt = pika::this_thread::experimental;

template <class T, Device D>
auto newBlockMatrixContiguous() {
  auto layout = matrix::colMajorLayout({13, 13}, {13, 13}, 13);
  auto dist = matrix::Distribution({13, 13}, {13, 13});

  auto matrix = matrix::Matrix<T, D>(dist, layout);

  EXPECT_TRUE(data_iscontiguous(common::make_data(matrix.read(LocalTileIndex(0, 0)).get())));

  return matrix;
}

template <class T, Device D>
auto newBlockMatrixStrided() {
  auto layout = matrix::colMajorLayout({13, 13}, {13, 13}, 26);
  auto dist = matrix::Distribution({13, 13}, {13, 13});

  auto matrix = matrix::Matrix<T, D>(dist, layout);

  EXPECT_FALSE(data_iscontiguous(common::make_data(matrix.read(LocalTileIndex(0, 0)).get())));

  return matrix;
}

enum class ContiguousInput : bool { Yes = true, No = false };

template <ContiguousInput contiguous_input, class T, Device input_device, Device destination_device,
          internal::CopyToDestination copy_to_destination,
          internal::CopyFromDestination copy_from_destination,
          internal::RequireContiguous require_contiguous>
void testWithTemporaryTile() {
  matrix::Matrix<T, input_device> matrix = bool(contiguous_input)
                                               ? newBlockMatrixContiguous<T, input_device>()
                                               : newBlockMatrixStrided<T, input_device>();

  const bool is_contiguous =
      tt::sync_wait(matrix.readwrite_sender(LocalTileIndex(0, 0))).is_contiguous();
  constexpr bool is_same_device = input_device == destination_device;
  const bool expect_new_tile = !is_same_device || (bool(require_contiguous) && !is_contiguous);
  const T* input_ptr = tt::sync_wait(matrix.readwrite_sender(LocalTileIndex(0, 0))).ptr();

  T input_value = 42.0;
  T output_value = 17.0;

  // If we expect the temporary tile to be a copy the pointers should be
  // different.
  auto check_tile_ptr = [&](const T* temp_ptr) {
    if (expect_new_tile) {
      EXPECT_NE(input_ptr, temp_ptr);
    }
    else {
      EXPECT_EQ(input_ptr, temp_ptr);
    }
  };
  auto check_tile_contiguous = [&](const auto& temp) {
    if (bool(require_contiguous)) {
      EXPECT_TRUE(temp.is_contiguous());
    }
  };

  auto check_input_value = [&](const TileElementIndex&) { return input_value; };
  auto check_output_value = [&](const TileElementIndex&) { return output_value; };

  // Test passing a read-write tile to withTemporaryTile.
  {
    // Synchronously set values in the tile.
    tt::sync_wait(internal::whenAllLift(blas::Uplo::General, input_value, input_value,
                                        matrix.readwrite_sender(LocalTileIndex(0, 0))) |
                  tile::laset(internal::Policy<DefaultBackend_v<input_device>>()));

    // Actually use withTemporaryTile, checking the values of the temporary
    // tiles asynchronously.
    auto sender = internal::withTemporaryTile<
        destination_device, copy_to_destination, copy_from_destination,
        require_contiguous>(matrix.readwrite_sender(LocalTileIndex(0, 0)), [&](auto& temp) {
      check_tile_ptr(temp.ptr());
      check_tile_contiguous(temp);

      ex::unique_any_sender<> check_tile_sender{ex::just()};
      if (bool(copy_to_destination) || !expect_new_tile) {
        // If the input tile should be copied to the temporary tile or if we
        // don't expect a copy the temporary tile should have the values set
        // above. This is done asynchronously since we don't know what context
        // we are currently running on and it may not be safe to call
        // sync_wait.
        check_tile_sender =
            ex::just(std::cref(temp)) |
            internal::transform(internal::Policy<
                                    matrix::internal::CopyBackend_v<destination_device, Device::CPU>>(),
                                matrix::Duplicate<Device::CPU>{}) |
            ex::then([&](const auto& tile_cpu) { CHECK_TILE_EQ(check_input_value, tile_cpu); });
      }

      return internal::whenAllLift(std::move(check_tile_sender), blas::Uplo::General, output_value,
                                   output_value, std::cref(temp)) |
             tile::laset(internal::Policy<DefaultBackend_v<destination_device>>());
    });
    static_assert(
        std::is_same_v<matrix::Tile<T, input_device>, decltype(tt::sync_wait(std::move(sender)))>);
    auto tile = tt::sync_wait(std::move(sender));

    // If the temporary tile should be copied back to the input or if we don't
    // expect a new tile the original tile should have the values set in
    // withTemporaryTile. Otherwise it should still have the input values.
    auto tile_cpu = tt::sync_wait(
        ex::just(std::cref(tile)) |
        internal::transform(internal::Policy<matrix::internal::CopyBackend_v<input_device, Device::CPU>>(),
                            matrix::Duplicate<Device::CPU>{}));
    if (bool(copy_from_destination) || !expect_new_tile) {
      CHECK_TILE_EQ(check_output_value, tile_cpu);
    }
    else {
      CHECK_TILE_EQ(check_input_value, tile_cpu);
    }
  }

  // Change the tested values since we are reusing the input matrix.
  input_value += 3.0;
  output_value += 3.0;

  // Test passing a read-only tile to withTemporaryTile. Copying back from the
  // temporary to the input can not be done with read-only access.
  if constexpr (!bool(copy_from_destination)) {
    // Synchronously set values in the tile.
    tt::sync_wait(internal::whenAllLift(blas::Uplo::General, input_value, input_value,
                                        matrix.readwrite_sender(LocalTileIndex(0, 0))) |
                  tile::laset(internal::Policy<DefaultBackend_v<input_device>>()));

    // Actually use withTemporaryTile, checking the values of the temporary
    // tiles asynchronously.
    auto sender = internal::withTemporaryTile<
        destination_device, copy_to_destination, copy_from_destination,
        require_contiguous>(matrix.read_sender(LocalTileIndex(0, 0)), [&](auto& temp) {
      check_tile_ptr(temp.ptr());
      check_tile_contiguous(temp);

      ex::unique_any_sender<> check_tile_sender{ex::just()};
      if (bool(copy_to_destination) || !expect_new_tile) {
        // If the input tile should be copied to the temporary tile or if we
        // don't expect a copy the temporary tile should have the values set
        // above. This is done asynchronously since we don't know what context
        // we are currently running on and it may not be safe to call
        // sync_wait.
        check_tile_sender =
            ex::just(std::cref(temp)) |
            internal::transform(internal::Policy<
                                    matrix::internal::CopyBackend_v<destination_device, Device::CPU>>(),
                                matrix::Duplicate<Device::CPU>{}) |
            ex::then([&](const auto& tile_cpu) { CHECK_TILE_EQ(check_input_value, tile_cpu); });
      }

      return check_tile_sender;
    });
    static_assert(std::is_void_v<decltype(tt::sync_wait(std::move(sender)))>);
    tt::sync_wait(std::move(sender));

    // When the input is read-only we can't write to it. The input tile should
    // still contain the values we set in the beginning.
    auto tile_cpu = tt::sync_wait(
        matrix.read_sender(LocalTileIndex(0, 0)) |
        internal::transform(internal::Policy<matrix::internal::CopyBackend_v<input_device, Device::CPU>>(),
                            matrix::Duplicate<Device::CPU>{}));
    CHECK_TILE_EQ(check_input_value, tile_cpu);
  }
}

template <ContiguousInput contiguous_input, Device destination_device, class T, Device input_device>
void testWithTemporaryTileHelper() {
  // clang-format off
  testWithTemporaryTile<contiguous_input, T, input_device, destination_device, internal::CopyToDestination::No, internal::CopyFromDestination::No, internal::RequireContiguous::No>();
  testWithTemporaryTile<contiguous_input, T, input_device, destination_device, internal::CopyToDestination::Yes, internal::CopyFromDestination::No, internal::RequireContiguous::No>();

  testWithTemporaryTile<contiguous_input, T, input_device, destination_device, internal::CopyToDestination::No, internal::CopyFromDestination::Yes, internal::RequireContiguous::No>();
  testWithTemporaryTile<contiguous_input, T, input_device, destination_device, internal::CopyToDestination::Yes, internal::CopyFromDestination::Yes, internal::RequireContiguous::No>();

  testWithTemporaryTile<contiguous_input, T, input_device, destination_device, internal::CopyToDestination::No, internal::CopyFromDestination::No, internal::RequireContiguous::Yes>();
  testWithTemporaryTile<contiguous_input, T, input_device, destination_device, internal::CopyToDestination::Yes, internal::CopyFromDestination::No, internal::RequireContiguous::Yes>();

  testWithTemporaryTile<contiguous_input, T, input_device, destination_device, internal::CopyToDestination::No, internal::CopyFromDestination::Yes, internal::RequireContiguous::Yes>();
  testWithTemporaryTile<contiguous_input, T, input_device, destination_device, internal::CopyToDestination::Yes, internal::CopyFromDestination::Yes, internal::RequireContiguous::Yes>();
  // clang-format on
}

using T = double;

TEST(WithTemporaryTile, Contiguous) {
  testWithTemporaryTileHelper<ContiguousInput::Yes, Device::CPU, T, Device::CPU>();
#ifdef DLAF_WITH_GPU
  testWithTemporaryTileHelper<ContiguousInput::Yes, Device::CPU, T, Device::GPU>();
  testWithTemporaryTileHelper<ContiguousInput::Yes, Device::GPU, T, Device::CPU>();
  testWithTemporaryTileHelper<ContiguousInput::Yes, Device::GPU, T, Device::GPU>();
#endif
}

TEST(WithTemporaryTile, Strided) {
  testWithTemporaryTileHelper<ContiguousInput::No, Device::CPU, T, Device::CPU>();
#ifdef DLAF_WITH_GPU
  testWithTemporaryTileHelper<ContiguousInput::No, Device::CPU, T, Device::GPU>();
  testWithTemporaryTileHelper<ContiguousInput::No, Device::GPU, T, Device::CPU>();
  testWithTemporaryTileHelper<ContiguousInput::No, Device::GPU, T, Device::GPU>();
#endif
}
