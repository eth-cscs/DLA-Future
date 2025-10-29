//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// @file

#include <variant>

#include <dlaf/matrix/allocation_types.h>
#include <dlaf/tune.h>
#include <dlaf/types.h>

#define MATRIXALLOCATIONSPEC_ASSERT_MESSAGE                                  \
  "Constructor signature is AllocationSpec(AllocationLayoutType, LdType);\n" \
  "All parameters are optional, but order is enforced."

namespace dlaf::matrix {

/// Manages the specs for matrix allocation.
class AllocationSpec {
public:
  using AllocationLayoutType = std::variant<AllocationLayoutDefault, AllocationLayout>;
  using LdType = std::variant<LdDefault, LdSpec>;

  ///@{
  /// Construct a AllocationSpec object
  ///
  /// @params layout (optional, default AllocationLayoutDefault{}) can be either of type
  ///         @p AllocationLayoutDefault or @p AllocationLayout (see layout() for more details).
  /// @params ld (optional, default LdDefault{}) can be either of type
  ///         @p LdDefault, @p LdSpec, @p Ld or @p SizeType (see ld() for more details).
  AllocationSpec() {}

  template <class... T>
  AllocationSpec(T... params) : AllocationSpec() {
    set<0>(params...);
  }
  ///@}

  /// Sets the spec for the leading dimension.
  ///
  /// See layout() for more details.
  AllocationSpec& set_layout(AllocationLayoutType layout) noexcept {
    layout_ = layout;
    return *this;
  }

  /// Sets the spec for the leading dimension.
  ///
  /// See ld() for more details.
  AllocationSpec& set_ld(LdType ld) noexcept {
    ld_ = ld;
    return *this;
  }

  /// Returns the spec for the leading dimension.
  ///
  /// If layout is set to default the default value set in tune parameters is returned.
  AllocationLayout layout() const noexcept {
    if (std::holds_alternative<AllocationLayoutDefault>(layout_)) {
      return getTuneParameters().default_allocation_layout;
    }
    return std::get<AllocationLayout>(layout_);
  }

  /// Returns the spec for the leading dimension.
  ///
  /// If ld is set to default:
  ///   - Ld::Padded is returned if layout() resolves to AllocationLayout::ColMajor
  ///   - Ld::Compact is returned in the other cases (AllocationLayout::Blocks, AllocationLayout::Tiles)
  LdSpec ld() const noexcept {
    if (std::holds_alternative<LdDefault>(ld_)) {
      if (layout() == AllocationLayout::ColMajor)
        return Ld::Padded;
      return Ld::Compact;
    }
    return std::get<LdSpec>(ld_);
  }

private:
  template <int>
  void set() {}

  template <int index, class... T>
  void set(AllocationLayoutType layout, T... params) {
    static_assert(index < 1, MATRIXALLOCATIONSPEC_ASSERT_MESSAGE);
    set_layout(layout);
    set<1>(params...);
  }
  template <int index, class... T>
  void set(LdType ld, T... params) {
    static_assert(index < 2, MATRIXALLOCATIONSPEC_ASSERT_MESSAGE);
    set_ld(ld);
    set<2>(params...);
  }

  AllocationLayoutType layout_ = AllocationLayoutDefault{};
  LdType ld_ = LdDefault{};
};
}

#undef MATRIXALLOCATIONSPEC_ASSERT_MESSAGE
