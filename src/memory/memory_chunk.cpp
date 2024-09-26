//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <cstddef>

#include <umpire/ResourceManager.hpp>
#include <umpire/strategy/PoolCoalesceHeuristic.hpp>
#include <umpire/strategy/QuickPool.hpp>
#include <umpire/strategy/ThreadSafeAllocator.hpp>

#include <dlaf/common/assert.h>
#include <dlaf/memory/memory_chunk.h>

namespace dlaf {
namespace memory {
namespace internal {
#ifdef DLAF_WITH_GPU
umpire::Allocator& getUmpireHostAllocator() {
  static auto host_allocator = umpire::ResourceManager::getInstance().getAllocator("PINNED");
  return host_allocator;
}

umpire::Allocator& getUmpireDeviceAllocator() {
  static auto device_allocator = umpire::ResourceManager::getInstance().getAllocator("DEVICE");
  return device_allocator;
}
#else
umpire::Allocator& getUmpireHostAllocator() {
  static auto host_allocator = umpire::ResourceManager::getInstance().getAllocator("HOST");
  return host_allocator;
}
#endif

using PoolType = umpire::strategy::QuickPool;
using CoalesceHeuristicType = umpire::strategy::PoolCoalesceHeuristic<PoolType>;

// This is a modified version of the "percent_releasable" coalescing heuristic
// from Umpire. This version allows choosing what ratio of the actual size to
// reallocate when coalescing.
//
// A free ratio of 1.0 means that the pool will be coalesced only when all
// blocks are unused. A free ratio of 0.5 means that the pool will be coalesced
// when at least 50% of the pool's memory is unused. A ratio of 0.0 means that
// the pool will be coalesced as soon as any two free blocks are available. A
// ratio of more than 1.0 will make the pool never coalesce.
//
// A reallocation ratio of 1.0 simply coalesces all the free memory into a new
// block. A ratio of 0.5 will attempt to shrink the pool to half its previous
// size. A ratio of 1.5 will allocate 50% more than the previous pool size.
//
// A single free block is never "coalesced" to keep things simple. In theory a
// single block could be shrunk or grown to match the reallocation ratio but
// this can lead to strange reallocations, so we simply avoid that case. Two or
// more blocks are always coalesced to one block, so no reallocation will
// happen immediately after coalescing two or more blocks.
static CoalesceHeuristicType get_coalesce_heuristic(double coalesce_free_ratio,
                                                    double coalesce_reallocation_ratio) {
  return [=](const PoolType& pool) {
    std::size_t threshold = static_cast<std::size_t>(coalesce_free_ratio * pool.getActualSize());
    if (pool.getReleasableBlocks() >= 2 && pool.getReleasableSize() >= threshold) {
      return static_cast<std::size_t>(coalesce_reallocation_ratio * pool.getActualSize());
    }
    else {
      return static_cast<std::size_t>(0);
    }
  };
}

void initializeUmpireHostAllocator(std::size_t initial_block_bytes, std::size_t next_block_bytes,
                                   std::size_t alignment_bytes, double coalesce_free_ratio,
                                   double coalesce_reallocation_ratio) {
#ifdef DLAF_WITH_GPU
  static bool initialized = false;

  // Umpire pools cannot be released, so we keep the pools around even when
  // DLA-Future is reinitialized.
  if (!initialized) {
    auto host_allocator = umpire::ResourceManager::getInstance().getAllocator("PINNED");
    auto pooled_host_allocator =
        umpire::ResourceManager::getInstance().makeAllocator<umpire::strategy::QuickPool>(
            "DLAF_PINNED_pool", host_allocator, initial_block_bytes, next_block_bytes, alignment_bytes,
            get_coalesce_heuristic(coalesce_free_ratio, coalesce_reallocation_ratio));
    auto thread_safe_pooled_host_allocator =
        umpire::ResourceManager::getInstance().makeAllocator<umpire::strategy::ThreadSafeAllocator>(
            "DLAF_PINNED_thread_safe_pool", pooled_host_allocator);

    memory::internal::getUmpireHostAllocator() = thread_safe_pooled_host_allocator;

    initialized = true;
  }
#else
  dlaf::internal::silenceUnusedWarningFor(initial_block_bytes, next_block_bytes, alignment_bytes,
                                          coalesce_free_ratio, coalesce_reallocation_ratio);
#endif
}

void finalizeUmpireHostAllocator() {}

#ifdef DLAF_WITH_GPU
void initializeUmpireDeviceAllocator(std::size_t initial_block_bytes, std::size_t next_block_bytes,
                                     std::size_t alignment_bytes, double coalesce_free_ratio,
                                     double coalesce_reallocation_ratio) {
  static bool initialized = false;

  // Umpire pools cannot be released, so we keep the pools around even when
  // DLA-Future is reinitialized.
  if (!initialized) {
    auto device_allocator = umpire::ResourceManager::getInstance().getAllocator("DEVICE");
    auto pooled_device_allocator =
        umpire::ResourceManager::getInstance().makeAllocator<umpire::strategy::QuickPool>(
            "DLAF_DEVICE_pool", device_allocator, initial_block_bytes, next_block_bytes, alignment_bytes,
            get_coalesce_heuristic(coalesce_free_ratio, coalesce_reallocation_ratio));
    auto thread_safe_pooled_device_allocator =
        umpire::ResourceManager::getInstance().makeAllocator<umpire::strategy::ThreadSafeAllocator>(
            "DLAF_DEVICE_thread_safe_pool", pooled_device_allocator);

    memory::internal::getUmpireDeviceAllocator() = thread_safe_pooled_device_allocator;

    initialized = true;
  }
}

void finalizeUmpireDeviceAllocator() {}
#endif
}
}
}
