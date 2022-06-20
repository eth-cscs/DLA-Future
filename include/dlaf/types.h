//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// @file

#include <complex>
#include <cstddef>
#include <limits>
#include <ostream>

#include "dlaf/common/assert.h"

namespace dlaf {

using SizeType = std::ptrdiff_t;

static_assert(std::is_signed_v<SizeType> && std::is_integral_v<SizeType>,
              "SizeType should be a signed integral type");
static_assert(sizeof(SizeType) >= 4, "SizeType should be >= 32bit");

enum class Device {
  CPU,
  GPU,
#ifdef DLAF_WITH_CUDA
  Default = GPU
#else
  Default = CPU
#endif
};

inline std::ostream& operator<<(std::ostream& os, const Device& device) {
  switch (device) {
    case Device::CPU:
      os << "CPU";
      break;
    case Device::GPU:
      os << "GPU";
      break;
  }
  return os;
}

enum class Backend {
  MC,
  GPU,
#ifdef DLAF_WITH_CUDA
  Default = GPU
#else
  Default = MC
#endif
};

inline std::ostream& operator<<(std::ostream& os, const Backend& backend) {
  switch (backend) {
    case Backend::MC:
      os << "MC";
      break;
    case Backend::GPU:
      os << "GPU";
      break;
  }
  return os;
}

/// Default device given a backend.
template <Backend backend>
struct DefaultDevice;

template <>
struct DefaultDevice<Backend::MC> {
  static constexpr Device value = Device::CPU;
};

template <>
struct DefaultDevice<Backend::GPU> {
  static constexpr Device value = Device::GPU;
};

template <Backend backend>
inline constexpr Device DefaultDevice_v = DefaultDevice<backend>::value;

/// Default backend given a device.
template <Device device>
struct DefaultBackend;

template <>
struct DefaultBackend<Device::CPU> {
  static constexpr Backend value = Backend::MC;
};

template <>
struct DefaultBackend<Device::GPU> {
  static constexpr Backend value = Backend::GPU;
};

template <Device device>
inline constexpr Backend DefaultBackend_v = DefaultBackend<device>::value;

template <class T>
struct TypeInfo;

template <class T>
struct TypeInfo<const T> : public TypeInfo<T> {};

template <class T>
struct TypeInfo {
  using BaseType = T;
  using Type = T;
  using ComplexType = std::complex<T>;

  static constexpr int ops_add = 1;
  static constexpr int ops_mul = 1;
};

template <class T>
struct TypeInfo<std::complex<T>> {
  using BaseType = T;
  using Type = std::complex<T>;
  using ComplexType = std::complex<T>;

  static constexpr int ops_add = 2;
  static constexpr int ops_mul = 6;
};

template <class T>
using BaseType = typename TypeInfo<T>::BaseType;

template <class T>
using ComplexType = typename TypeInfo<T>::ComplexType;

template <class T>
inline constexpr bool isComplex_v = std::is_same_v<T, ComplexType<T>>;

/// Compute the number of operations.
///
/// Given the number of additions and multiplications of type @tparam T,
/// it returns the number of basic floating point operations.
template <class T>
constexpr double total_ops(const double add, const double mul) {
  return TypeInfo<T>::ops_add * add + TypeInfo<T>::ops_mul * mul;
}

/// Return complex conjugate of a complex number.
template <class T>
std::complex<T> conj(const std::complex<T> number) {
  return std::conj(number);
}

/// Return complex conjugate of a real number as a real number.
///
/// It differs from std::conj just in the return type. In fact,
/// std::conj always returns a std::complex.
template <class T>
T conj(const T number) {
  return number;
}

/// Cast from unsigned to signed integer types.
///
/// It performs the cast checking if the given unsigned value can be stored in the destination type.
template <class S, class U,
          std::enable_if_t<std::is_integral_v<U> && std::is_unsigned_v<U> && std::is_integral_v<S> &&
                               std::is_signed_v<S>,
                           int> = 0>
S to_signed(const U unsigned_value) {
  DLAF_ASSERT_MODERATE(static_cast<std::size_t>(std::numeric_limits<S>::max()) >= unsigned_value,
                       std::numeric_limits<S>::max(), unsigned_value);
  return static_cast<S>(unsigned_value);
}

/// Fallback.
template <class S, class SB,
          std::enable_if_t<std::is_integral_v<SB> && std::is_signed_v<SB> && std::is_integral_v<S> &&
                               std::is_signed_v<S>,
                           int> = 0>
S to_signed(const SB value) {
  DLAF_ASSERT_MODERATE(std::numeric_limits<S>::max() >= value, std::numeric_limits<S>::max(), value);
  DLAF_ASSERT_MODERATE(std::numeric_limits<S>::min() <= value, std::numeric_limits<S>::min(), value);
  return static_cast<S>(value);
}

/// Cast from signed to unsigned integer types.
///
/// It performs the cast checking if the given signed value is greater than 0 and if the destination type
/// can store the value.
template <class U, class S,
          std::enable_if_t<std::is_integral_v<U> && std::is_unsigned_v<U> && std::is_integral_v<S> &&
                               std::is_signed_v<S>,
                           int> = 0>
U to_unsigned(const S signed_value) {
  DLAF_ASSERT_MODERATE(signed_value >= 0, signed_value);
  DLAF_ASSERT_MODERATE(std::numeric_limits<U>::max() >= static_cast<std::size_t>(signed_value),
                       std::numeric_limits<U>::max(), static_cast<std::size_t>(signed_value));
  return static_cast<U>(signed_value);
}

/// Fallback.
template <class U, class UB,
          std::enable_if_t<std::is_integral_v<U> && std::is_unsigned_v<U> && std::is_integral_v<UB> &&
                               std::is_unsigned_v<UB>,
                           int> = 0>
U to_unsigned(const UB unsigned_value) {
  DLAF_ASSERT_MODERATE(std::numeric_limits<U>::max() >= static_cast<std::size_t>(unsigned_value),
                       std::numeric_limits<U>::max(), static_cast<std::size_t>(unsigned_value));
  return static_cast<U>(unsigned_value);
}

template <class To, class From,
          std::enable_if_t<std::is_integral_v<From> && std::is_integral_v<To> && std::is_unsigned_v<To>,
                           int> = 0>
To integral_cast(const From value) {
  return to_unsigned<To, From>(value);
}

template <
    class To, class From,
    std::enable_if_t<std::is_integral_v<From> && std::is_integral_v<To> && std::is_signed_v<To>, int> = 0>
To integral_cast(const From value) {
  return to_signed<To, From>(value);
}

/// Helper function for casting from any integer type to int.
///
/// Useful when passing parameters to the MPI interface,
/// see dlaf::to_signed.
template <class T>
auto to_int(const T unsigned_value) {
  return to_signed<int>(unsigned_value);
}

/// Helper function for casting from any integer type to unsigned int.
///
/// Useful when creating dim3 objects for CUDA,
/// see dlaf::to_unsigned.
template <class T>
auto to_uint(const T unsigned_value) {
  return to_unsigned<unsigned>(unsigned_value);
}

/// Helper function for casting from any integer type to std::size_t.
///
/// Useful for interaction between std, but not only, with other interfaces that does not use unsigned
/// types (e.g. MPI, BLAS, ...) see dlaf::to_unsigned.
template <class T>
auto to_sizet(const T signed_value) {
  return to_unsigned<std::size_t>(signed_value);
}

/// Helper function for casting from any integer type to dlaf::SizeType.
///
/// Useful when passing parameter to the BLAS/LAPACK interface,
/// see dlaf::to_signed.
template <class T>
auto to_SizeType(const T unsigned_value) {
  return to_signed<SizeType>(unsigned_value);
}
}
