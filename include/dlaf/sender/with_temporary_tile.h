//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <functional>
#include <utility>

#include <pika/execution.hpp>

#include <dlaf/common/unwrap.h>
#include <dlaf/matrix/copy_tile.h>
#include <dlaf/sender/policy.h>
#include <dlaf/types.h>

namespace dlaf::internal {
/// This represents whether or not withTemporaryTile should copy the input tile
/// to the destination before the user-supplied operation runs.
enum class CopyToDestination : bool { Yes = true, No = false };

/// This represents whether or not withTemporaryTile should copy the temporary
/// tile to the input after the user-supplied operation has run.
enum class CopyFromDestination : bool { Yes = true, No = false };

/// This represents whether withTemporaryTile requires that the temporary tile
/// uses contiguous memory.
enum class RequireContiguous : bool { Yes = true, No = false };

template <typename T>
struct moveNonConstTile {
  T& tile;
  auto operator()() {
    if constexpr (!std::is_const_v<std::remove_reference_t<T>>) {
      return std::move(tile);
    }
  }
};

template <typename T>
moveNonConstTile(T&) -> moveNonConstTile<T>;

/// This is a sender adaptor that takes a sender sending a tile, and gives
/// access by reference to a temporary or the input tile depending on
/// compile-time options.
///
/// This sender adaptor provides access to a temporary tile, if certain
/// conditions are met. Fundamentally it tries to avoid allocating a new tile if
/// it is not required.
///
/// If the requested destination_device for the temporary tile is different from
/// the device of the input tile a new tile will be allocated.  If the user
/// additionally requests the temporary tile must use contiguous memory a
/// runtime check will be performed on the input tile, and if it is not
/// contiguous a new tile will also be allocated.
///
/// In addition to requesting a destination device and contiguous memory, the
/// user may request additional operations that will be performed only when a
/// new tile is allocated. If the input tile is used directly none of the
/// following operations will be performed as they are unnecessary. If
/// copy_to_destination is CopyToDestination::Yes the input tile will first be
/// copied to the temporary tile before calling the user provided callable f. If
/// copy_from_destination is CopyFromDestination::Yes the temporary tile will be
/// copied back to the input tile after the sender returned from the
/// user-provided callable f completes.
///
/// The adaptor returns a sender that sends nothing if the input tile contains
/// const elements, and the input tile if it contains non-const elements.
///
/// The operation roughly looks like the following diagrammatically. If a
/// temporary tile isn't required the flow is simple:
///
///   in_sender ---> f ---> returned sender
///
/// If a temporary tile is created:
///
///   in_sender ---> duplicate -----------------------------------------------------/---> returned sender
///                      |                                                          |
///                      \---> copy to temporary ---> f ---> copy from temporary ---/
///                                (optional)                      (optional)
template <Device destination_device, CopyToDestination copy_to_destination,
          CopyFromDestination copy_from_destination, RequireContiguous require_contiguous,
          typename InSender, typename F>
auto withTemporaryTile(InSender&& in_sender, F&& f) {
  namespace ex = pika::execution::experimental;

  using dlaf::internal::Policy;
  using dlaf::internal::transform;
  using dlaf::internal::whenAllLift;
  using dlaf::matrix::copy;
  using dlaf::matrix::Duplicate;
  using dlaf::matrix::DuplicateNoCopy;
  using dlaf::matrix::internal::CopyBackend_v;
  using pika::execution::thread_priority;

  return std::forward<InSender>(in_sender) |
         // Start a new asynchronous scope for keeping the input tile alive
         // until all asynchronous operations are done.
         ex::let_value(dlaf::common::internal::Unwrapping{[f = std::forward<F>(f)](auto& in) mutable {
           constexpr Device in_device_type = std::decay_t<decltype(in)>::device;
           constexpr Backend copy_backend = CopyBackend_v<in_device_type, destination_device>;

           // In cases that we cannot use the input tile as the temporary tile we need to:
           // 1. allocate a new tile
           // 2. optionally copy the input to the temporary tile
           // 3. call the user-provided callable f and ignore values sent by it
           // 4. optionally copy the temporary tile back to the input tile
           // 5. send the input tile to continuations
           auto helper_withtemp = [&]() mutable {
             // If the user requested copying the input tile to the temporary
             // tile we use Duplicate and copy_backend. If the user did not
             // request copying to the temporary tile we still need to allocate
             // a new temporary tile with the correct dimensions. In that case
             // we always use DuplicateNoCopy on the MC backend since allocation
             // does not require invoking a kernel on a GPU.
             using duplicate_type =
                 std::conditional_t<bool(copy_to_destination), Duplicate<destination_device>,
                                    DuplicateNoCopy<destination_device>>;
             constexpr auto duplicate_backend = bool(copy_to_destination) ? copy_backend : Backend::MC;

             return ex::just(std::cref(in)) |
                    // Allocate a new tile and optionally copy the input tile to
                    // the temporary tile.
                    transform(Policy<duplicate_backend>(thread_priority::high), duplicate_type{}) |
                    // Start a new asynchronous scope for keeping the temporary
                    // tile alive until all asynchronous operations are done.
                    ex::let_value([&, f = std::forward<F>(f)](auto& temp) mutable {
                      // Call the user provided callable f and ignore the values
                      // sent by the sender.
                      auto f_sender = f(temp) | ex::drop_value();
                      // If the user requested copying the temporary tile back
                      // to the input tile after the user-provided operation is
                      // done we do so. Otherwise we use the sender f_sender
                      // directly.
                      auto copy_sender = [&]() {
                        if constexpr (bool(copy_from_destination)) {
                          return ex::make_unique_any_sender(
                              whenAllLift(std::move(f_sender), std::cref(temp), std::cref(in)) |
                              copy(Policy<copy_backend>(thread_priority::high)));
                        }
                        else {
                          return ex::make_unique_any_sender(std::move(f_sender));
                        }
                      }();
                      // Send the input tile to continuations if the tile is
                      // non-const.
                      return std::move(copy_sender) | ex::then(moveNonConstTile{in});
                    });
           };

           // One of the helpers may be unused depending on which branch is taken
           dlaf::internal::silenceUnusedWarningFor(helper_withtemp);

           // If the destination device is the same as the input device we may
           // be able to avoid allocating a new tile.
           if constexpr (in_device_type == destination_device) {
             // In some cases we can directly use the input as a temporary tile.
             // In that case we simply call the user-provided callable f, ignore
             // the values sent by the sender returned from f, and finally send
             // the input tile to continuations if the tile is non-const. This
             // helper is inside the if constexpr to avoid instantiating it in
             // cases where it's never needed.
             auto helper_notemp = [&]() {
               return f(in) | ex::drop_value() | ex::then(moveNonConstTile{in});
             };

             // The destination device is the same as the input device, but if
             // we require that the temporary tile is contiguous we may have to
             // allocate a new tile in any case. We have to do a runtime check
             // to find out. Since this is a runtime check, we wrap the senders
             // from the different branches in type-erased unique_any_senders.
             if constexpr (require_contiguous == RequireContiguous::Yes) {
               // If the input tile is contiguous we can use the input tile as a
               // temporary tile directly.
               if (in.is_contiguous()) {
                 return ex::make_unique_any_sender(helper_notemp());
               }
               // If the input is not contiguous we have to at least allocate a
               // new temporary tile.
               else {
                 return ex::make_unique_any_sender(helper_withtemp());
               }
             }
             // The destination device is the same as the input device, and we
             // don't require contiguous memory (though we allow it). We can use
             // the input tile as a temporary tile directly.
             else {
               return helper_notemp();
             }
           }
           // If the destination device is different from the input device we
           // have to at least allocate a new temporary tile on the destination
           // device.
           else {
             return helper_withtemp();
           }
         }});
}
}
