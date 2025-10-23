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

std::vector<AllocationLayout> layouts = {AllocationLayout::ColMajor, AllocationLayout::Blocks,
                                         AllocationLayout::Tiles};
std::vector<Ld> lds = {Ld::Compact, Ld::Padded};
std::vector<SizeType> ld_ints = {3, 7, 8};
std::vector<LdSpec> ldspecs = {Ld::Compact, 5};

TEST(AllocationSpecTest, DefaultConstructor) {
  AllocationSpec alloc;
  EXPECT_TRUE(test_default_layout_ld(alloc));
}

TEST(AllocationSpecTest, ConstructorLayout) {
  AllocationSpec alloc(AllocationLayoutDefault{});
  EXPECT_TRUE(test_default_layout_ld(alloc));

  for (const auto& layout : layouts) {
    AllocationSpec alloc(layout);
    EXPECT_TRUE(test_layout(alloc, layout));
    EXPECT_TRUE(test_default_ld(alloc));
  }
}

TEST(AllocationSpecTest, ConstructorLd) {
  AllocationSpec alloc(LdDefault{});
  EXPECT_TRUE(test_default_layout_ld(alloc));

  for (const auto& ld : lds) {
    AllocationSpec alloc(ld);
    EXPECT_TRUE(test_default_layout(alloc));
    EXPECT_TRUE(test_ld(alloc, ld));
  }
  for (const auto& ld : ld_ints) {
    AllocationSpec alloc(ld);
    EXPECT_TRUE(test_default_layout(alloc));
    EXPECT_TRUE(test_ld(alloc, ld));
  }
  for (const auto& ld : ldspecs) {
    AllocationSpec alloc(ld);
    EXPECT_TRUE(test_default_layout(alloc));
    EXPECT_TRUE(test_ld(alloc, ld));
  }
}

TEST(AllocationSpecTest, ConstructorLayoutLd) {
  AllocationSpec alloc(AllocationLayoutDefault{}, LdDefault{});
  EXPECT_TRUE(test_default_layout_ld(alloc));

  AllocationSpec alloc1(AllocationLayoutDefault{}, LdSpec{5});
  EXPECT_TRUE(test_default_layout(alloc1));
  EXPECT_TRUE(test_ld(alloc1, 5));

  AllocationSpec alloc2(AllocationLayout::ColMajor, LdDefault{});
  EXPECT_TRUE(test_layout(alloc2, AllocationLayout::ColMajor));
  EXPECT_TRUE(test_default_ld(alloc2));

  for (const auto& layout : layouts) {
    for (const auto& ld : lds) {
      AllocationSpec alloc(layout, ld);
      EXPECT_TRUE(test_layout(alloc, layout));
      EXPECT_TRUE(test_ld(alloc, ld));
    }
    for (const auto& ld : ld_ints) {
      AllocationSpec alloc(layout, ld);
      EXPECT_TRUE(test_layout(alloc, layout));
      EXPECT_TRUE(test_ld(alloc, ld));
    }
    for (const auto& ld : ldspecs) {
      AllocationSpec alloc(layout, ld);
      EXPECT_TRUE(test_layout(alloc, layout));
      EXPECT_TRUE(test_ld(alloc, ld));
    }
  }
}

TEST(AllocationSpecTest, SetLayout) {
  AllocationSpec alloc;
  ASSERT_TRUE(test_default_layout_ld(alloc));

  for (const auto& layout : layouts) {
    alloc.set_layout(layout);
    EXPECT_TRUE(test_layout(alloc, layout));
    EXPECT_TRUE(test_default_ld(alloc));
  }

  alloc.set_layout(AllocationLayoutDefault{});
  EXPECT_TRUE(test_default_layout_ld(alloc));
}

TEST(AllocationSpecTest, SetLd) {
  AllocationSpec alloc;
  ASSERT_TRUE(test_default_layout_ld(alloc));

  for (const auto& ld : lds) {
    alloc.set_ld(ld);
    EXPECT_TRUE(test_default_layout(alloc));
    EXPECT_TRUE(test_ld(alloc, ld));
  }
  for (const auto& ld : ld_ints) {
    alloc.set_ld(ld);
    EXPECT_TRUE(test_default_layout(alloc));
    EXPECT_TRUE(test_ld(alloc, ld));
  }
  for (const auto& ld : ldspecs) {
    alloc.set_ld(ld);
    EXPECT_TRUE(test_default_layout(alloc));
    EXPECT_TRUE(test_ld(alloc, ld));
  }

  alloc.set_ld(LdDefault{});
  EXPECT_TRUE(test_default_layout_ld(alloc));
}

TEST(AllocationSpecTest, SetConcatenation) {
  AllocationSpec alloc;
  alloc.set_layout(AllocationLayout::ColMajor).set_ld(5).set_ld(Ld::Padded);
  EXPECT_TRUE(test_layout(alloc, AllocationLayout::ColMajor));
  EXPECT_TRUE(test_ld(alloc, Ld::Padded));
}
