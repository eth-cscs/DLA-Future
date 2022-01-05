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

#include <hpx/program_options.hpp>

#include <dlaf/common/assert.h>
#include <dlaf/types.h>

#include <exception>
#include <iostream>
#include <string>

namespace dlaf::miniapp {
Backend parse_backend(const std::string& backend) {
  if (backend == "default")
    return Backend::Default;
  else if (backend == "mc")
    return Backend::MC;
  else if (backend == "gpu") {
#if !defined(DLAF_WITH_CUDA)
    std::cout << "Asked for --backend=gpu but DLAF_WITH_CUDA is disabled!" << std::endl;
    std::terminate();
#endif
    return Backend::GPU;
  }

  std::cout << "Parsing is not implemented for --backend=" << backend << "!" << std::endl;
  std::terminate();
  return Backend::Default;  // unreachable
}

enum class ElementType { Single, Double, ComplexSingle, ComplexDouble };

ElementType parse_element_type(const std::string& type) {
  if (type == "single")
    return ElementType::Single;
  else if (type == "double")
    return ElementType::Double;
  else if (type == "complex-single")
    return ElementType::ComplexSingle;
  else if (type == "complex-double")
    return ElementType::ComplexDouble;

  std::cout << "Parsing is not implemented for --type=" << type << "!" << std::endl;
  std::terminate();
  return ElementType::Single;  // unreachable
}

inline std::ostream& operator<<(std::ostream& os, const ElementType& type) {
  if (type == ElementType::Single) {
    os << "single";
  }
  else if (type == ElementType::Double) {
    os << "double";
  }
  else if (type == ElementType::ComplexSingle) {
    os << "complex-single";
  }
  else if (type == ElementType::ComplexDouble) {
    os << "complex-double";
  }
  return os;
}

blas::Uplo parse_uplo(const std::string& uplo) {
  if (uplo == "lower")
    return blas::Uplo::Lower;
  else if (uplo == "upper")
    return blas::Uplo::Upper;
  else if (uplo == "general")
    return blas::Uplo::General;

  std::cout << "Parsing is not implemented for --uplo=" << uplo << "!" << std::endl;
  std::terminate();
  return blas::Uplo::Lower;  // unreachable
}

struct MiniappOptions {
  Backend backend;
  ElementType type;
  int64_t nruns;
  int64_t nwarmups;

  MiniappOptions(const hpx::program_options::variables_map& vm)
      : backend(parse_backend(vm["backend"].as<std::string>())),
        type(parse_element_type(vm["type"].as<std::string>())), nruns(vm["nruns"].as<int64_t>()),
        nwarmups(vm["nwarmups"].as<int64_t>()) {
    DLAF_ASSERT(nruns > 0, nruns);
    DLAF_ASSERT(nwarmups >= 0, nwarmups);
  }

  MiniappOptions(MiniappOptions&&) = default;
  MiniappOptions(const MiniappOptions&) = default;
  MiniappOptions& operator=(MiniappOptions&&) = default;
  MiniappOptions& operator=(const MiniappOptions&) = default;
};

inline hpx::program_options::options_description getMiniappOptionsDescription() {
  hpx::program_options::options_description desc("DLA-Future miniapp options");

  desc.add_options()("nruns", hpx::program_options::value<int64_t>()->default_value(1),
                     "Number of runs");
  desc.add_options()("nwarmups", hpx::program_options::value<int64_t>()->default_value(1),
                     "Number of warmup runs");
  desc.add_options()(
      "backend", hpx::program_options::value<std::string>()->default_value("default"),
      "Backend to use ('default' ('gpu' if available, otherwise 'mc'), 'mc', 'gpu' (if available))");
  desc.add_options()("type", hpx::program_options::value<std::string>()->default_value("double"),
                     "Element type to use ('single', 'double', 'complex-single', 'complex-double')");

  return desc;
}
}
