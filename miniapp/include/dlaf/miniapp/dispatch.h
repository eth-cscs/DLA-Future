//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <dlaf/types.h>
#include <dlaf/miniapp/options.h>

#include <exception>
#include <iostream>

namespace dlaf::miniapp {
namespace detail {
template <typename Miniapp, Backend backend, typename OptionsType>
void dispatch_miniapp_element_type(const OptionsType& opts) {
  switch (opts.type) {
    case ElementType::Single:
      Miniapp::template run<backend, float>(opts);
      break;
    case ElementType::Double:
      Miniapp::template run<backend, double>(opts);
      break;
    case ElementType::ComplexSingle:
      Miniapp::template run<backend, std::complex<float>>(opts);
      break;
    case ElementType::ComplexDouble:
      Miniapp::template run<backend, std::complex<double>>(opts);
      break;
    default:
      std::cout << "Unknown element type given (" << opts.type << ")!" << std::endl;
      std::terminate();
  }
}

template <typename Miniapp, typename OptionsType>
void dispatch_miniapp_backend(const OptionsType& opts) {
  switch (opts.backend) {
    case Backend::MC:
      dispatch_miniapp_element_type<Miniapp, Backend::MC>(opts);
      break;
    case Backend::GPU:
#ifdef DLAF_WITH_CUDA
      dispatch_miniapp_element_type<Miniapp, Backend::MC>(opts);
#else
      std::cout << "Attempting to dispatch to Backend::GPU but DLAF_WITH_CUDA is disabled!" << std::endl;
      std::terminate();
#endif
      break;
    default:
      std::cout << "Unknown backend given (" << opts.backend << ")!" << std::endl;
      std::terminate();
  }
}
}

template <typename Miniapp, typename OptionsType>
void dispatch_miniapp(const OptionsType& opts) {
  detail::dispatch_miniapp_backend<Miniapp>(opts);
}
}
