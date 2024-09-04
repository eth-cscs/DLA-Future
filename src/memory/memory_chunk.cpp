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

#include <mpi.h>

#include <umpire/ResourceManager.hpp>
#include <umpire/strategy/QuickPool.hpp>
#include <umpire/strategy/ThreadSafeAllocator.hpp>

#include <pika/init.hpp>

#include <dlaf/memory/memory_chunk.h>

#include <string_view>

namespace dlaf {
namespace memory {
namespace internal {
static void print_alloc_stats(std::string label, std::ostream& os) {
  auto alloc = umpire::ResourceManager::getInstance().getAllocator(label);
  os << "name: " << alloc.getName() << ", ";
  os << "id: " << alloc.getId() << ", ";
  os << "strategy: " << alloc.getStrategyName() << ", ";
  os << "high water: " << alloc.getHighWatermark() << ", ";
  os << "current size: " << alloc.getCurrentSize() << ", ";
  os << "actual size: " << alloc.getActualSize() << ", ";
  os << "alloc count: " << alloc.getAllocationCount() << ", ";
  os << '\n';
}

void print_cuda_stats(std::string_view label) {
  // if (pika::is_runtime_initialized()) { pika::wait(); }
  // cudaDeviceSynchronize();
  std::size_t cuda_free, cuda_total;
  int id = 0;
  cudaGetDevice(&id);
  cudaMemGetInfo(&cuda_free, &cuda_total);

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::ostringstream os;
  os << "### " << label << '\n';
  os << "rank: " << rank << ", ";
  os << "cuda_free: " << cuda_free << ", ";
  os << "cuda_total: " << cuda_total << '\n';

  print_alloc_stats("DEVICE", os);
  print_alloc_stats("DEVICE_pool", os);
  print_alloc_stats("DEVICE_thread_safe_pool", os);
  print_alloc_stats("PINNED", os);
  print_alloc_stats("PINNED_pool", os);
  print_alloc_stats("PINNED_thread_safe_pool", os);

  std::cerr << os.str();
}

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

void initializeUmpireHostAllocator(std::size_t initial_bytes) {
#ifdef DLAF_WITH_GPU
  static bool initialized = false;

  // Umpire pools cannot be released, so we keep the pools around even when
  // DLA-Future is reinitialized.
  if (!initialized) {
    auto host_allocator = umpire::ResourceManager::getInstance().getAllocator("PINNED");
    auto pooled_host_allocator =
        umpire::ResourceManager::getInstance().makeAllocator<umpire::strategy::QuickPool>("PINNED_pool",
                                                                                          host_allocator,
                                                                                          initial_bytes);
    auto thread_safe_pooled_host_allocator =
        umpire::ResourceManager::getInstance().makeAllocator<umpire::strategy::ThreadSafeAllocator>(
            "PINNED_thread_safe_pool", pooled_host_allocator);

    memory::internal::getUmpireHostAllocator() = thread_safe_pooled_host_allocator;

    initialized = true;
  }
#else
  (void) initial_bytes;
#endif
}

void finalizeUmpireHostAllocator() {}

#ifdef DLAF_WITH_GPU
void initializeUmpireDeviceAllocator(std::size_t initial_bytes) {
  static bool initialized = false;

  // Umpire pools cannot be released, so we keep the pools around even when
  // DLA-Future is reinitialized.
  if (!initialized) {
    auto device_allocator = umpire::ResourceManager::getInstance().getAllocator("DEVICE");
    auto pooled_device_allocator =
        umpire::ResourceManager::getInstance().makeAllocator<umpire::strategy::QuickPool>(
            "DEVICE_pool", device_allocator, initial_bytes);
    auto thread_safe_pooled_device_allocator =
        umpire::ResourceManager::getInstance().makeAllocator<umpire::strategy::ThreadSafeAllocator>(
            "DEVICE_thread_safe_pool", pooled_device_allocator);

    memory::internal::getUmpireDeviceAllocator() = thread_safe_pooled_device_allocator;

    initialized = true;
  }
}

void finalizeUmpireDeviceAllocator() {}
#endif
}
}
}
