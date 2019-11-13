#include "dlaf/common/pipeline.h"

#include <gtest/gtest.h>

#include <hpx/lcos/future.hpp>

using namespace dlaf;
using dlaf::common::Pipeline;

TEST(Pipeline, Basic) {
  int resource = 26;
  Pipeline<int> serial(std::move(resource));

  auto checkpoint0 = serial();
  auto checkpoint1 = checkpoint0.then(hpx::util::unwrapping([](auto&& wrapper) {
    return std::move(wrapper);
  }));

  auto guard0 = serial();
  auto guard1 = serial();

  EXPECT_TRUE(checkpoint1.is_ready());
  EXPECT_FALSE(guard0.is_ready());
  EXPECT_FALSE(guard1.is_ready());

  checkpoint1.get();

  EXPECT_TRUE(guard0.is_ready());
  EXPECT_FALSE(guard1.is_ready());

  guard0.get();

  EXPECT_TRUE(guard1.is_ready());

  guard1.get();
}
