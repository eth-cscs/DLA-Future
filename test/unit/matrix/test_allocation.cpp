//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <variant>
#include <vector>

#include <dlaf/matrix/allocation.h>
#include <dlaf/tune.h>

#include <gtest/gtest.h>

using dlaf::getTuneParameters;
using dlaf::SizeType;
using dlaf::matrix::AllocationLayout;
using dlaf::matrix::AllocationLayoutDefault;
using dlaf::matrix::AllocationSpec;
using dlaf::matrix::Ld;
using dlaf::matrix::LdDefault;
using dlaf::matrix::LdSpec;
using dlaf::memory::AllocateOn;
using dlaf::memory::AllocateOnDefault;

bool test_default_layout(const AllocationSpec& alloc) {
  bool flag = true;
  getTuneParameters().default_allocation_layout = AllocationLayout::ColMajor;
  flag = flag && alloc.layout() == AllocationLayout::ColMajor;
  getTuneParameters().default_allocation_layout = AllocationLayout::Blocks;
  flag = flag && alloc.layout() == AllocationLayout::Blocks;
  getTuneParameters().default_allocation_layout = AllocationLayout::Tiles;
  flag = flag && alloc.layout() == AllocationLayout::Tiles;
  return flag;
}

bool test_default_layout_ld(const AllocationSpec& alloc) {
  if (!std::holds_alternative<Ld>(alloc.ld()))
    return false;
  bool flag = true;
  getTuneParameters().default_allocation_layout = AllocationLayout::ColMajor;
  flag = flag && alloc.layout() == AllocationLayout::ColMajor;
  flag = flag && std::get<Ld>(alloc.ld()) == Ld::Padded;
  getTuneParameters().default_allocation_layout = AllocationLayout::Blocks;
  flag = flag && alloc.layout() == AllocationLayout::Blocks;
  flag = flag && std::get<Ld>(alloc.ld()) == Ld::Compact;
  getTuneParameters().default_allocation_layout = AllocationLayout::Tiles;
  flag = flag && alloc.layout() == AllocationLayout::Tiles;
  flag = flag && std::get<Ld>(alloc.ld()) == Ld::Compact;
  return flag;
}

bool test_default_ld(const AllocationSpec& alloc) {
  if (!std::holds_alternative<Ld>(alloc.ld()))
    return false;
  bool flag = true;
  AllocationSpec copy(alloc);
  copy.set_layout(AllocationLayout::ColMajor);
  flag = flag && std::get<Ld>(copy.ld()) == Ld::Padded;
  copy.set_layout(AllocationLayout::Blocks);
  flag = flag && std::get<Ld>(copy.ld()) == Ld::Compact;
  copy.set_layout(AllocationLayout::Tiles);
  flag = flag && std::get<Ld>(copy.ld()) == Ld::Compact;
  return flag;
}

bool test_default_allocate_on(const AllocationSpec& alloc) {
  bool flag = true;
  getTuneParameters().default_allocate_on = AllocateOn::Construction;
  flag = flag && alloc.allocate_on() == AllocateOn::Construction;
  getTuneParameters().default_allocate_on = AllocateOn::Demand;
  flag = flag && alloc.allocate_on() == AllocateOn::Demand;
  return flag;
}

bool test_layout(const AllocationSpec& alloc, AllocationLayout layout) {
  if (layout == AllocationLayout::ColMajor)
    getTuneParameters().default_allocation_layout = AllocationLayout::Blocks;
  else
    getTuneParameters().default_allocation_layout = AllocationLayout::ColMajor;
  return alloc.layout() == layout;
}

bool test_ld(const AllocationSpec& alloc, SizeType ld) {
  if (!std::holds_alternative<SizeType>(alloc.ld()))
    return false;
  return std::get<SizeType>(alloc.ld()) == ld;
}
bool test_ld(const AllocationSpec& alloc, Ld ld) {
  if (!std::holds_alternative<Ld>(alloc.ld()))
    return false;
  if (ld == Ld::Padded)
    getTuneParameters().default_allocation_layout = AllocationLayout::Blocks;
  else
    getTuneParameters().default_allocation_layout = AllocationLayout::ColMajor;
  return std::get<Ld>(alloc.ld()) == ld;
}
bool test_ld(const AllocationSpec& alloc, LdSpec ld) {
  return std::visit([&alloc](const auto& ld) { return test_ld(alloc, ld); }, ld);
}
bool test_allocate_on(const AllocationSpec& alloc, AllocateOn allocate_on) {
  if (allocate_on == AllocateOn::Construction)
    getTuneParameters().default_allocate_on = AllocateOn::Demand;
  else
    getTuneParameters().default_allocate_on = AllocateOn::Construction;
  return alloc.allocate_on() == allocate_on;
}

std::vector<AllocationLayout> layouts = {AllocationLayout::ColMajor, AllocationLayout::Blocks,
                                         AllocationLayout::Tiles};
std::vector<Ld> lds = {Ld::Compact, Ld::Padded};
std::vector<SizeType> ld_ints = {3, 7, 8};
std::vector<LdSpec> ldspecs = {Ld::Compact, 5};

std::vector<AllocateOn> allocate_ons = {AllocateOn::Construction, AllocateOn::Demand};

TEST(AllocationSpecTest, DefaultConstructor) {
  AllocationSpec alloc;
  EXPECT_TRUE(test_default_layout_ld(alloc));
  EXPECT_TRUE(test_default_allocate_on(alloc));
}

TEST(AllocationSpecTest, ConstructorLayout) {
  AllocationSpec alloc(AllocationLayoutDefault{});
  EXPECT_TRUE(test_default_layout_ld(alloc));
  EXPECT_TRUE(test_default_allocate_on(alloc));

  for (const auto& layout : layouts) {
    AllocationSpec alloc(layout);
    EXPECT_TRUE(test_layout(alloc, layout));
    EXPECT_TRUE(test_default_ld(alloc));
    EXPECT_TRUE(test_default_allocate_on(alloc));
  }
}

TEST(AllocationSpecTest, ConstructorLd) {
  AllocationSpec alloc(LdDefault{});
  EXPECT_TRUE(test_default_layout_ld(alloc));
  EXPECT_TRUE(test_default_allocate_on(alloc));

  for (const auto& ld : lds) {
    AllocationSpec alloc(ld);
    EXPECT_TRUE(test_default_layout(alloc));
    EXPECT_TRUE(test_ld(alloc, ld));
    EXPECT_TRUE(test_default_allocate_on(alloc));
  }
  for (const auto& ld : ld_ints) {
    AllocationSpec alloc(ld);
    EXPECT_TRUE(test_default_layout(alloc));
    EXPECT_TRUE(test_ld(alloc, ld));
    EXPECT_TRUE(test_default_allocate_on(alloc));
  }
  for (const auto& ld : ldspecs) {
    AllocationSpec alloc(ld);
    EXPECT_TRUE(test_default_layout(alloc));
    EXPECT_TRUE(test_ld(alloc, ld));
    EXPECT_TRUE(test_default_allocate_on(alloc));
  }
}

TEST(AllocationSpecTest, ConstructorAllocateOn) {
  AllocationSpec alloc(AllocateOnDefault{});
  ASSERT_TRUE(test_default_layout_ld(alloc));
  ASSERT_TRUE(test_default_allocate_on(alloc));

  for (const auto& allocate_on : allocate_ons) {
    AllocationSpec alloc(allocate_on);
    EXPECT_TRUE(test_default_layout_ld(alloc));
    EXPECT_TRUE(test_allocate_on(alloc, allocate_on));
  }
}

TEST(AllocationSpecTest, ConstructorLayoutLd) {
  AllocationSpec alloc(AllocationLayoutDefault{}, LdDefault{});
  EXPECT_TRUE(test_default_layout_ld(alloc));
  EXPECT_TRUE(test_default_allocate_on(alloc));

  AllocationSpec alloc1(AllocationLayoutDefault{}, LdSpec{5});
  EXPECT_TRUE(test_default_layout(alloc1));
  EXPECT_TRUE(test_ld(alloc1, 5));
  EXPECT_TRUE(test_default_allocate_on(alloc1));

  AllocationSpec alloc2(AllocationLayout::ColMajor, LdDefault{});
  EXPECT_TRUE(test_layout(alloc2, AllocationLayout::ColMajor));
  EXPECT_TRUE(test_default_ld(alloc2));
  EXPECT_TRUE(test_default_allocate_on(alloc2));

  for (const auto& layout : layouts) {
    for (const auto& ld : lds) {
      AllocationSpec alloc(layout, ld);
      EXPECT_TRUE(test_layout(alloc, layout));
      EXPECT_TRUE(test_ld(alloc, ld));
      EXPECT_TRUE(test_default_allocate_on(alloc));
    }
    for (const auto& ld : ld_ints) {
      AllocationSpec alloc(layout, ld);
      EXPECT_TRUE(test_layout(alloc, layout));
      EXPECT_TRUE(test_ld(alloc, ld));
      EXPECT_TRUE(test_default_allocate_on(alloc));
    }
    for (const auto& ld : ldspecs) {
      AllocationSpec alloc(layout, ld);
      EXPECT_TRUE(test_layout(alloc, layout));
      EXPECT_TRUE(test_ld(alloc, ld));
      EXPECT_TRUE(test_default_allocate_on(alloc));
    }
  }
}

TEST(AllocationSpecTest, ConstructorLayoutAllocateOn) {
  AllocationSpec alloc(AllocationLayoutDefault{}, AllocateOnDefault{});
  EXPECT_TRUE(test_default_layout_ld(alloc));
  EXPECT_TRUE(test_default_allocate_on(alloc));

  AllocationSpec alloc1(AllocationLayoutDefault{}, AllocateOn::Construction);
  EXPECT_TRUE(test_default_layout_ld(alloc1));
  EXPECT_TRUE(test_allocate_on(alloc1, AllocateOn::Construction));

  AllocationSpec alloc2(AllocationLayout::ColMajor, AllocateOnDefault{});
  EXPECT_TRUE(test_layout(alloc2, AllocationLayout::ColMajor));
  EXPECT_TRUE(test_default_ld(alloc2));
  EXPECT_TRUE(test_default_allocate_on(alloc2));

  for (const auto& layout : layouts) {
    for (const auto& allocate_on : allocate_ons) {
      AllocationSpec alloc(layout, allocate_on);
      EXPECT_TRUE(test_layout(alloc, layout));
      EXPECT_TRUE(test_default_ld(alloc));
      EXPECT_TRUE(test_allocate_on(alloc, allocate_on));
    }
  }
}

TEST(AllocationSpecTest, ConstructorLdAllocateOn) {
  AllocationSpec alloc(LdDefault{}, AllocateOnDefault{});
  EXPECT_TRUE(test_default_layout_ld(alloc));
  EXPECT_TRUE(test_default_allocate_on(alloc));

  AllocationSpec alloc1(LdDefault{}, AllocateOn::Construction);
  EXPECT_TRUE(test_default_layout_ld(alloc1));
  EXPECT_TRUE(test_allocate_on(alloc1, AllocateOn::Construction));

  AllocationSpec alloc2(LdSpec{5}, AllocateOnDefault{});
  EXPECT_TRUE(test_default_layout(alloc2));
  EXPECT_TRUE(test_ld(alloc2, 5));
  EXPECT_TRUE(test_default_allocate_on(alloc2));

  for (const auto& allocate_on : allocate_ons) {
    for (const auto& ld : lds) {
      AllocationSpec alloc(ld, allocate_on);
      EXPECT_TRUE(test_default_layout(alloc));
      EXPECT_TRUE(test_ld(alloc, ld));
      EXPECT_TRUE(test_allocate_on(alloc, allocate_on));
    }
    for (const auto& ld : ld_ints) {
      AllocationSpec alloc(ld, allocate_on);
      EXPECT_TRUE(test_default_layout(alloc));
      EXPECT_TRUE(test_ld(alloc, ld));
      EXPECT_TRUE(test_allocate_on(alloc, allocate_on));
    }
    for (const auto& ld : ldspecs) {
      AllocationSpec alloc(ld, allocate_on);
      EXPECT_TRUE(test_default_layout(alloc));
      EXPECT_TRUE(test_ld(alloc, ld));
      EXPECT_TRUE(test_allocate_on(alloc, allocate_on));
    }
  }
}

TEST(AllocationSpecTest, ConstructorLayoutLdAllocateOn) {
  AllocationSpec alloc(AllocationLayoutDefault{}, LdDefault{}, AllocateOnDefault{});
  EXPECT_TRUE(test_default_layout_ld(alloc));
  EXPECT_TRUE(test_default_allocate_on(alloc));

  AllocationSpec alloc1(AllocationLayoutDefault{}, LdSpec{5}, AllocateOn::Construction);
  EXPECT_TRUE(test_default_layout(alloc1));
  EXPECT_TRUE(test_ld(alloc1, 5));
  EXPECT_TRUE(test_allocate_on(alloc1, AllocateOn::Construction));

  AllocationSpec alloc2(AllocationLayout::ColMajor, LdDefault{}, AllocateOn::Construction);
  EXPECT_TRUE(test_layout(alloc2, AllocationLayout::ColMajor));
  EXPECT_TRUE(test_default_ld(alloc2));
  EXPECT_TRUE(test_allocate_on(alloc2, AllocateOn::Construction));

  AllocationSpec alloc3(AllocationLayout::ColMajor, LdSpec{5}, AllocateOnDefault{});
  EXPECT_TRUE(test_layout(alloc3, AllocationLayout::ColMajor));
  EXPECT_TRUE(test_ld(alloc3, 5));
  EXPECT_TRUE(test_default_allocate_on(alloc3));

  for (const auto& layout : layouts) {
    for (const auto& allocate_on : allocate_ons) {
      for (const auto& ld : lds) {
        AllocationSpec alloc(layout, ld, allocate_on);
        EXPECT_TRUE(test_layout(alloc, layout));
        EXPECT_TRUE(test_ld(alloc, ld));
        EXPECT_TRUE(test_allocate_on(alloc, allocate_on));
      }
      for (const auto& ld : ld_ints) {
        AllocationSpec alloc(layout, ld, allocate_on);
        EXPECT_TRUE(test_layout(alloc, layout));
        EXPECT_TRUE(test_ld(alloc, ld));
        EXPECT_TRUE(test_allocate_on(alloc, allocate_on));
      }
      for (const auto& ld : ldspecs) {
        AllocationSpec alloc(layout, ld, allocate_on);
        EXPECT_TRUE(test_layout(alloc, layout));
        EXPECT_TRUE(test_ld(alloc, ld));
        EXPECT_TRUE(test_allocate_on(alloc, allocate_on));
      }
    }
  }
}

TEST(AllocationSpecTest, SetLayout) {
  AllocationSpec alloc;
  ASSERT_TRUE(test_default_layout_ld(alloc));
  ASSERT_TRUE(test_default_allocate_on(alloc));

  for (const auto& layout : layouts) {
    alloc.set_layout(layout);
    EXPECT_TRUE(test_layout(alloc, layout));
    EXPECT_TRUE(test_default_ld(alloc));
    EXPECT_TRUE(test_default_allocate_on(alloc));
  }

  alloc.set_layout(AllocationLayoutDefault{});
  EXPECT_TRUE(test_default_layout_ld(alloc));
  EXPECT_TRUE(test_default_allocate_on(alloc));
  EXPECT_TRUE(test_default_allocate_on(alloc));
}

TEST(AllocationSpecTest, SetLd) {
  AllocationSpec alloc;
  ASSERT_TRUE(test_default_layout_ld(alloc));
  ASSERT_TRUE(test_default_allocate_on(alloc));

  for (const auto& ld : lds) {
    alloc.set_ld(ld);
    EXPECT_TRUE(test_default_layout(alloc));
    EXPECT_TRUE(test_ld(alloc, ld));
    EXPECT_TRUE(test_default_allocate_on(alloc));
  }
  for (const auto& ld : ld_ints) {
    alloc.set_ld(ld);
    EXPECT_TRUE(test_default_layout(alloc));
    EXPECT_TRUE(test_ld(alloc, ld));
    EXPECT_TRUE(test_default_allocate_on(alloc));
  }
  for (const auto& ld : ldspecs) {
    alloc.set_ld(ld);
    EXPECT_TRUE(test_default_layout(alloc));
    EXPECT_TRUE(test_ld(alloc, ld));
    EXPECT_TRUE(test_default_allocate_on(alloc));
  }

  alloc.set_ld(LdDefault{});
  EXPECT_TRUE(test_default_layout_ld(alloc));
  EXPECT_TRUE(test_default_allocate_on(alloc));
}

TEST(AllocationSpecTest, SetAllocateOn) {
  AllocationSpec alloc;
  ASSERT_TRUE(test_default_layout_ld(alloc));
  ASSERT_TRUE(test_default_allocate_on(alloc));

  for (const auto& allocate_on : allocate_ons) {
    alloc.set_allocate_on(allocate_on);
    EXPECT_TRUE(test_default_layout_ld(alloc));
    EXPECT_TRUE(test_allocate_on(alloc, allocate_on));
  }

  alloc.set_allocate_on(AllocateOnDefault{});
  EXPECT_TRUE(test_default_layout_ld(alloc));
  EXPECT_TRUE(test_default_allocate_on(alloc));
}

TEST(AllocationSpecTest, SetConcatenation) {
  AllocationSpec alloc;
  alloc.set_layout(AllocationLayout::ColMajor).set_ld(5).set_ld(Ld::Padded);
  EXPECT_TRUE(test_layout(alloc, AllocationLayout::ColMajor));
  EXPECT_TRUE(test_ld(alloc, Ld::Padded));
}
