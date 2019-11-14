#include <hpx/hpx_init.hpp>

#include "dlaf/common/pool.h"

using dlaf::common::Pool;

TEST(Pool, Basic) {
  Pool<int, 2> pool;

  auto step1 = pool.get();
  auto step2 = pool.get();
  auto step3 = pool.get();

  step1.get()() = 13;

  std::cout << step2.get()() << std::endl;
  std::cout << step3.get()() << std::endl;

  return hpx::finalize();
}
