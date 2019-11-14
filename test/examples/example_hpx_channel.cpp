#include <hpx/hpx_init.hpp>

#include "dlaf/common/pool.h"

using dlaf::common::Pool;

int hpx_main(int argc, char* argv[]) {
  Pool<int, 2> pool;

  auto step1 = pool.get();
  auto step2 = pool.get();
  auto step3 = pool.get();

  step1.get().get_value() = 13;

  std::cout << step2.get().get_value() << std::endl;
  std::cout << step3.get().get_value() << std::endl;

  return hpx::finalize();
}

int main(int argc, char* argv[]) {
  return hpx::init(argc, argv);
}
