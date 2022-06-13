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

#include <dlaf/matrix/copy_tile.h>
#include <dlaf/sender/policy.h>
#include <dlaf/types.h>

namespace dlaf::comm::internal {
// Needed helpers:
//
// send: copy into buffer -> comm -> discard
//
// reduceSend: copy into contiguous buffer -> comm -> discard
//
// recv: comm                  --> copy into buffer
//       create similar buffer -/  // this comes first because we need the size of the input
//
// reduceRecvInPlace: same as recv, require contiguous
//

// This one is currently used for recvBcast. It creates a duplicated tile on the CPU.
//
// It creates a "similar" tile on the communication device, if needed. It does
// not copy the contents of the tile to the communication tile.
//
// TODO: Name.
template <typename InSender, typename F>
auto with_similar_comm_tile(InSender&& in_sender, F&& f) {
  namespace ex = pika::execution::experimental;
  return std::forward<InSender>(in_sender) | ex::let_value([f = std::forward<F>(f)](auto& in) mutable {
           constexpr Device in_device_type = std::decay_t<decltype(in)>::D;
           constexpr Device comm_device_type = CommunicationDevice<in_device_type>::value;

           if constexpr (in_device_type == comm_device_type) {
             return f(in, in);
           }
           else {
             const dlaf::internal::Policy<Backend::MC> policy{/* TODO: priority */};
             return ex::just(std::cref(in)) |
                    transform(policy, dlaf::matrix::DuplicateNoCopy<comm_device_type>{}) |
                    ex::let_value(
                        [&in, f = std::forward<F>(f)](auto& out) mutable { return f(in, out); });
           }
         });
}

// This one is currently used for sendBcast. It copies the tile to the communication device first and
// then it performs the communication.
template <typename InSender, typename F>
auto with_comm_tile(InSender&& in_sender, F&& f) {
  namespace ex = pika::execution::experimental;
  return std::forward<InSender>(in_sender) |
         ex::let_value(pika::unwrapping([f = std::forward<F>(f)](auto& in) mutable {
           constexpr Device in_device_type = std::decay_t<decltype(in)>::D;
           constexpr Device comm_device_type = CommunicationDevice<in_device_type>::value;
           constexpr Backend copy_backend =
               dlaf::matrix::internal::CopyBackend_v<in_device_type, comm_device_type>;

           if constexpr (in_device_type == comm_device_type) {
             return f(in, in);
           }
           else {
             const dlaf::internal::Policy<copy_backend> policy{/* TODO: priority */};
             return ex::just(std::cref(in)) |
                    dlaf::internal::transform(policy, dlaf::matrix::Duplicate<comm_device_type>{}) |
                    ex::let_value(
                        [&in, f = std::forward<F>(f)](auto& comm) mutable { return f(in, comm); });
           }
         }));
}

// TODO: Move to pika (also add make_any_sender).
template <typename Sender>
auto make_unique_any_sender(Sender&& sender) {
  using value_types_pack = typename pika::execution::experimental::sender_traits<
      std::decay_t<Sender>>::template value_types<pika::util::pack, pika::util::pack>;
  using single_value_type_variant =
      pika::execution::experimental::detail::single_variant_t<value_types_pack>;
  using unique_any_sender_type =
      pika::util::detail::change_pack_t<pika::execution::experimental::unique_any_sender,
                                        single_value_type_variant>;

  return unique_any_sender_type(std::forward<Sender>(sender));
}

// This one is currently used for reduceRecvInPlace. It first creates a
// "similar" tile on the communication device if needed. It ensures that the
// communication tile is contiguous.
template <typename InSender, typename F>
auto with_similar_contiguous_comm_tile(InSender&& in_sender, F&& f) {
  namespace ex = pika::execution::experimental;
  return std::forward<InSender>(in_sender) |
         ex::let_value(pika::unwrapping([f = std::forward<F>(f)](auto& in) mutable {
           constexpr Device in_device_type = std::decay_t<decltype(in)>::D;
           constexpr Device comm_device_type = CommunicationDevice<in_device_type>::value;

           static_assert(comm_device_type == Device::CPU);

           if (in_device_type == comm_device_type && in.is_contiguous()) {
             return make_unique_any_sender(f(in, in));
           }
           else {
             const dlaf::internal::Policy<Backend::MC> policy{/* TODO: priority */};
             return make_unique_any_sender(
                 ex::just(std::cref(in)) |
                 dlaf::internal::transform(policy, dlaf::matrix::DuplicateNoCopy<comm_device_type>{}) |
                 ex::let_value(
                     [&in, f = std::forward<F>(f)](auto& comm) mutable { return f(in, comm); }));
           }
         }));
}

// This one is currently used for reduceSend. It copies the tile to the
// communication device first and then it performs the communication. It ensures
// that the communication tile is contiguous.
template <typename InSender, typename F>
auto with_contiguous_comm_tile(InSender&& in_sender, F&& f) {
  namespace ex = pika::execution::experimental;
  return std::forward<InSender>(in_sender) |
         ex::let_value(pika::unwrapping([f = std::forward<F>(f)](auto& in) mutable {
           constexpr Device in_device_type = std::decay_t<decltype(in)>::D;
           constexpr Device comm_device_type = CommunicationDevice<in_device_type>::value;
           constexpr Backend copy_backend =
               dlaf::matrix::internal::CopyBackend_v<in_device_type, comm_device_type>;

           if (in_device_type == comm_device_type && in.is_contiguous()) {
             return make_unique_any_sender(f(in, in));
           }
           else {
             const dlaf::internal::Policy<copy_backend> policy{/* TODO: priority */};
             return make_unique_any_sender(
                 ex::just(std::cref(in)) |
                 dlaf::internal::transform(policy, dlaf::matrix::Duplicate<comm_device_type>{}) |
                 ex::let_value(
                     [&in, f = std::forward<F>(f)](auto& comm) mutable { return f(in, comm); }));
           }
         }));
}

template <typename Sender, typename TileIn, typename TileContigComm>
auto copyBack(Sender&& sender, const TileIn& tile_in, const TileContigComm& tile_contig_comm) {
  using dlaf::internal::Policy;
  using dlaf::internal::whenAllLift;
  using dlaf::matrix::copy;
  using pika::threads::thread_priority;

  // operator== for Tile (the below is not 100% accurate if we have views)?
  if (tile_in.ptr() == tile_contig_comm.ptr()) {
    return make_unique_any_sender(std::forward<Sender>(sender));
  }
  else {
    constexpr Device in_device_type = std::decay_t<decltype(tile_in)>::D;
    constexpr Device comm_device_type = std::decay_t<decltype(tile_contig_comm)>::D;
    constexpr Backend copy_backend =
        dlaf::matrix::internal::CopyBackend_v<in_device_type, comm_device_type>;

    return make_unique_any_sender(
        whenAllLift(std::forward<Sender>(sender), std::cref(tile_contig_comm), std::cref(tile_in)) |
        copy(Policy<copy_backend>(thread_priority::high)));
  }
}
}
