//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

/// @file

#pragma once

#include <complex>

#include <dlaf/types.h>

#ifdef DLAF_WITH_GPU
#define DLAF_EXPAND_ETI_SDCZ_DEVICE(ETI_MACRO, KWORD)  \
  ETI_MACRO(KWORD, float, Device::CPU);                \
  ETI_MACRO(KWORD, double, Device::CPU);               \
  ETI_MACRO(KWORD, std::complex<float>, Device::CPU);  \
  ETI_MACRO(KWORD, std::complex<double>, Device::CPU); \
  ETI_MACRO(KWORD, float, Device::GPU);                \
  ETI_MACRO(KWORD, double, Device::GPU);               \
  ETI_MACRO(KWORD, std::complex<float>, Device::GPU);  \
  ETI_MACRO(KWORD, std::complex<double>, Device::GPU);
#define DLAF_EXPAND_ETI_SDCZ_DEVICE_VA_ARGS(ETI_MACRO, KWORD, ...)  \
  ETI_MACRO(KWORD, float, Device::CPU, __VA_ARGS__);                \
  ETI_MACRO(KWORD, double, Device::CPU, __VA_ARGS__);               \
  ETI_MACRO(KWORD, std::complex<float>, Device::CPU, __VA_ARGS__);  \
  ETI_MACRO(KWORD, std::complex<double>, Device::CPU, __VA_ARGS__); \
  ETI_MACRO(KWORD, float, Device::GPU, __VA_ARGS__);                \
  ETI_MACRO(KWORD, double, Device::GPU, __VA_ARGS__);               \
  ETI_MACRO(KWORD, std::complex<float>, Device::GPU, __VA_ARGS__);  \
  ETI_MACRO(KWORD, std::complex<double>, Device::GPU, __VA_ARGS__);
#else
#define DLAF_EXPAND_ETI_SDCZ_DEVICE(ETI_MACRO, KWORD) \
  ETI_MACRO(KWORD, float, Device::CPU);               \
  ETI_MACRO(KWORD, double, Device::CPU);              \
  ETI_MACRO(KWORD, std::complex<float>, Device::CPU); \
  ETI_MACRO(KWORD, std::complex<double>, Device::CPU);
#define DLAF_EXPAND_ETI_SDCZ_DEVICE_VA_ARGS(ETI_MACRO, KWORD, ...) \
  ETI_MACRO(KWORD, float, Device::CPU, __VA_ARGS__);               \
  ETI_MACRO(KWORD, double, Device::CPU, __VA_ARGS__);              \
  ETI_MACRO(KWORD, std::complex<float>, Device::CPU, __VA_ARGS__); \
  ETI_MACRO(KWORD, std::complex<double>, Device::CPU, __VA_ARGS__);
#endif
