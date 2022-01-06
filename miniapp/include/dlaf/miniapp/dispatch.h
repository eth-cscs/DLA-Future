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

#include <dlaf/miniapp/options.h>
#include <dlaf/types.h>

#include <complex>
#include <exception>
#include <iostream>

namespace dlaf::miniapp {
namespace internal {
template <typename Miniapp, Backend backend, typename OptionsType>
void dispatchMiniappElementType(const OptionsType& opts) {
  if (opts.type == ElementType::Single) {
    Miniapp::template run<backend, float>(opts);
    return;
  }
  else if (opts.type == ElementType::Double) {
    Miniapp::template run<backend, double>(opts);
    return;
  }
  else {
    if constexpr (OptionsType::supported_types == SupportedTypes::RealAndComplex) {
      if (opts.type == ElementType::ComplexSingle) {
        Miniapp::template run<backend, std::complex<float>>(opts);
        return;
      }
      else if (opts.type == ElementType::ComplexDouble) {
        Miniapp::template run<backend, std::complex<double>>(opts);
        return;
      }
    }
  }

  std::cout << "Unknown or unsupported type (" << opts.type << ")!" << std::endl;
  std::terminate();
}

template <typename Miniapp, typename OptionsType>
void dispatchMiniappBackend(const OptionsType& opts) {
  switch (opts.backend) {
    case Backend::MC:
      dispatchMiniappElementType<Miniapp, Backend::MC>(opts);
      break;
    case Backend::GPU:
#ifdef DLAF_WITH_CUDA
      dispatchMiniappElementType<Miniapp, Backend::GPU>(opts);
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
void dispatchMiniapp(const OptionsType& opts) {
  internal::dispatchMiniappBackend<Miniapp>(opts);
}
}
