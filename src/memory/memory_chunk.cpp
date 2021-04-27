#include <dlaf/memory/memory_chunk.h>

#include <umpire/ResourceManager.hpp>
#include <umpire/strategy/DynamicPool.hpp>
#include <umpire/strategy/ThreadSafeAllocator.hpp>

namespace dlaf {
namespace memory {
namespace internal {
#ifdef DLAF_WITH_CUDA
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
#ifdef DLAF_WITH_CUDA
  static bool initialized = false;

  // Umpire pools cannot be released, so we keep the pools around even when
  // DLA-Future is reinitialized.
  if (!initialized) {
    auto host_allocator = umpire::ResourceManager::getInstance().getAllocator("PINNED");
    auto pooled_host_allocator =
        umpire::ResourceManager::getInstance()
            .makeAllocator<umpire::strategy::DynamicPool>("PINNED_pool", host_allocator, initial_bytes);
    auto thread_safe_pooled_host_allocator =
        umpire::ResourceManager::getInstance()
            .makeAllocator<umpire::strategy::ThreadSafeAllocator>("PINNED_thread_safe_pool",
                                                                  pooled_host_allocator);

    memory::internal::getUmpireHostAllocator() = thread_safe_pooled_host_allocator;

    initialized = true;
  }
#endif
}

void finalizeUmpireHostAllocator() {}

#ifdef DLAF_WITH_CUDA
void initializeUmpireDeviceAllocator(std::size_t initial_bytes) {
  static bool initialized = false;

  // Umpire pools cannot be released, so we keep the pools around even when
  // DLA-Future is reinitialized.
  if (!initialized) {
    auto device_allocator = umpire::ResourceManager::getInstance().getAllocator("DEVICE");
    auto pooled_device_allocator =
        umpire::ResourceManager::getInstance()
            .makeAllocator<umpire::strategy::DynamicPool>("DEVICE_pool", device_allocator,
                                                          initial_bytes);
    auto thread_safe_pooled_device_allocator =
        umpire::ResourceManager::getInstance()
            .makeAllocator<umpire::strategy::ThreadSafeAllocator>("DEVICE_thread_safe_pool",
                                                                  pooled_device_allocator);

    memory::internal::getUmpireDeviceAllocator() = thread_safe_pooled_device_allocator;

    initialized = true;
  }
}

void finalizeUmpireDeviceAllocator() {}
#endif
}
}
}
