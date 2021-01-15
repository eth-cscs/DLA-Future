#include <dlaf/memory/memory_chunk.h>

#ifdef DLAF_WITH_UMPIRE
#include <umpire/ResourceManager.hpp>
#include <umpire/strategy/DynamicPool.hpp>
#include <umpire/strategy/ThreadSafeAllocator.hpp>

namespace dlaf {
namespace memory {
#ifdef DLAF_WITH_CUDA
// TODO: Initial pool size must be configurable.
umpire::Allocator& getHostAllocator() {
  static auto host_allocator = umpire::ResourceManager::getInstance().getAllocator("PINNED");
  static auto pooled_host_allocator =
      umpire::ResourceManager::getInstance().makeAllocator<umpire::strategy::DynamicPool>("PINNED_pool",
                                                                                          host_allocator,
                                                                                          4ul << 30);
  static auto thread_safe_pooled_host_allocator =
      umpire::ResourceManager::getInstance()
          .makeAllocator<umpire::strategy::ThreadSafeAllocator>("PINNED_thread_safe_pool",
                                                                pooled_host_allocator);
  return thread_safe_pooled_host_allocator;
}

umpire::Allocator& getDeviceAllocator() {
  static auto device_allocator = umpire::ResourceManager::getInstance().getAllocator("DEVICE");
  static auto pooled_device_allocator =
      umpire::ResourceManager::getInstance()
          .makeAllocator<umpire::strategy::DynamicPool>("DEVICE_pool", device_allocator, 4ul << 30);
  static auto thread_safe_pooled_device_allocator =
      umpire::ResourceManager::getInstance()
          .makeAllocator<umpire::strategy::ThreadSafeAllocator>("DEVICE_thread_safe_pool",
                                                                pooled_device_allocator);
  return thread_safe_pooled_device_allocator;
}
#else
umpire::Allocator& getHostAllocator() {
  static auto host_allocator = umpire::ResourceManager::getInstance().getAllocator("HOST");
  static auto pooled_host_allocator =
      umpire::ResourceManager::getInstance().makeAllocator<umpire::strategy::DynamicPool>("HOST_pool",
                                                                                          host_allocator,
                                                                                          4ul << 30);
  static auto thread_safe_pooled_host_allocator =
      umpire::ResourceManager::getInstance()
          .makeAllocator<umpire::strategy::ThreadSafeAllocator>("HOST_thread_safe_pool",
                                                                pooled_host_allocator);
  return thread_safe_pooled_host_allocator;
}
#endif
}
}

#endif
