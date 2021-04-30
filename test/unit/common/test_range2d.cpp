//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/common/index2d.h"
#include "dlaf/common/range2d.h"

#include <gtest/gtest.h>
#include <vector>

namespace {

struct CustomTag;
using Index = dlaf::common::Index2D<int, CustomTag>;
using Size = dlaf::common::Size2D<int, CustomTag>;
using dlaf::common::iterate_range2d;

// TypeParam is either `Index` or `Size`
template <typename TypeParam>
void test_single_arg() {
  TypeParam sz(7, 2);
  const auto range = iterate_range2d(sz);

  std::vector<Index> exp_values = {
      Index(0, 0), Index(1, 0), Index(2, 0), Index(3, 0), Index(4, 0), Index(5, 0), Index(6, 0),
      Index(0, 1), Index(1, 1), Index(2, 1), Index(3, 1), Index(4, 1), Index(5, 1), Index(6, 1),
  };

  std::vector<Index> act_values;
  act_values.reserve(exp_values.size());
  for (Index i : range) {
    act_values.push_back(i);
  }

  EXPECT_EQ(exp_values.size(), std::distance(range.begin(), range.end()));
  EXPECT_EQ(act_values, exp_values);
}

// `end` is either `Index(7, 4)` or `Size(4, 2)`
template <typename TypeParam>
void test_double_arg(TypeParam end) {
  Index begin(3, 2);
  const auto range = iterate_range2d(begin, end);

  EXPECT_EQ(3, range.begin()->row());
  EXPECT_EQ(2, range.begin()->col());

  std::vector<Index> exp_values = {Index(3, 2), Index(4, 2), Index(5, 2), Index(6, 2),
                                   Index(3, 3), Index(4, 3), Index(5, 3), Index(6, 3)};

  std::vector<Index> act_values;
  act_values.reserve(exp_values.size());
  for (Index i : range) {
    act_values.push_back(i);
  }

  EXPECT_EQ(exp_values.size(), std::distance(range.begin(), range.end()));
  EXPECT_EQ(act_values, exp_values);
}

}

TEST(SingleArgRange2D, Size2D) {
  ::test_single_arg<::Size>();
}

TEST(SingleArgRange2D, Index2D) {
  ::test_single_arg<::Index>();
}

TEST(DoubleArgRange2D, Index2D) {
  ::test_double_arg(Index(7, 4));
}

TEST(DoubleArgRange2D, Size2D) {
  ::test_double_arg(Size(4, 2));
}

template <class TypeParam>
void test_single_arg_empty() {
  TypeParam sz(0, 0);
  const auto range = iterate_range2d(sz);

  EXPECT_EQ(range.begin(), range.end());
}

template <class TypeParam>
void test_double_arg_empty(TypeParam end) {
  const ::Index begin(3, 2);
  const auto range = iterate_range2d(begin, end);

  EXPECT_EQ(range.begin(), range.end());
}

TEST(SingleArgEmptyRange2D, Size2D) {
  ::test_single_arg_empty<::Size>();
}

TEST(SingleArgEmptyRange2D, Index2D) {
  ::test_single_arg_empty<::Index>();
}

TEST(DoubleArgEmptyRange2D, Index2D) {
  ::test_double_arg_empty(Index(3, 2));
}

TEST(DoubleArgEmptyRange2D, Size2D) {
  ::test_double_arg_empty(Size(0, 0));
}
