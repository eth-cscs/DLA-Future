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

#include <algorithm>
#include <cctype>
#include <exception>
#include <iostream>
#include <string>
#include <type_traits>

#include <blas.hh>

#include <hpx/program_options.hpp>

#include <dlaf/common/assert.h>
#include <dlaf/common/format_short.h>
#include <dlaf/types.h>

namespace dlaf::miniapp {
inline Backend parseBackend(const std::string& backend) {
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

  std::cout << "Invalid option for --backend. Got '" << backend
            << "' but expected one of 'default', 'mc', 'gpu'." << std::endl;
  std::terminate();
  return Backend::Default;  // unreachable
}

enum class SupportReal { No, Yes };
enum class SupportComplex { No, Yes };
enum class ElementType { Single, Double, ComplexSingle, ComplexDouble };

template <SupportReal support_real, SupportComplex support_complex>
ElementType parseElementType(const std::string& type) {
  if (type.size() == 1) {
    const auto type_lower = std::tolower(type[0]);

    if constexpr (support_real == SupportReal::Yes) {
      if (type_lower == 's')
        return ElementType::Single;
      else if (type_lower == 'd')
        return ElementType::Double;
    }
    else {
      if (type_lower == 's' || type_lower == 'd') {
        std::cout << "--type=" << type << " is not supported in this miniapp!" << std::endl;
        std::terminate();
      }
    }

    if constexpr (support_complex == SupportComplex::Yes) {
      if (type_lower == 'c')
        return ElementType::ComplexSingle;
      else if (type_lower == 'z')
        return ElementType::ComplexDouble;
    }
    else {
      if (type_lower == 'c' || type_lower == 'z') {
        std::cout << "--type=" << type << " is not supported in this miniapp!" << std::endl;
        std::terminate();
      }
    }
  }

  std::cout << "Invalid option for --type. Got '" << type << "' but expected one of 's', 'd', 'c', 'z'."
            << std::endl;
  std::terminate();
  return ElementType::Single;  // unreachable
}

inline std::ostream& operator<<(std::ostream& os, const ElementType& type) {
  switch (type) {
    case ElementType::Single:
      os << "Single";
      break;
    case ElementType::Double:
      os << "Double";
      break;
    case ElementType::ComplexSingle:
      os << "ComplexSingle";
      break;
    case ElementType::ComplexDouble:
      os << "ComplexDouble";
      break;
    default:
      os << "unknown type (" << static_cast<std::underlying_type_t<ElementType>>(type) << ")";
  }
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const dlaf::internal::FormatShort<ElementType>& type) {
  switch (type.value) {
    case ElementType::Single:
      os << "s";
      break;
    case ElementType::Double:
      os << "d";
      break;
    case ElementType::ComplexSingle:
      os << "c";
      break;
    case ElementType::ComplexDouble:
      os << "z";
      break;
    default:
      os << "?";
  }
  return os;
}

enum class CheckIterFreq { None, Last, All };

inline CheckIterFreq parseCheckIterFreq(const std::string& check) {
  if (check == "all")
    return CheckIterFreq::All;
  else if (check == "last")
    return CheckIterFreq::Last;
  else if (check == "none")
    return CheckIterFreq::None;

  std::cout << "Parsing is not implemented for --check-result=" << check << "!" << std::endl;
  std::terminate();
  return CheckIterFreq::None;  // unreachable
}

namespace internal {
// This is a helper function for converting a command line option value (as a
// string) into a blaspp enum value. It assumes that the first character can be
// directly cast to the given enum type.
template <typename T>
T stringToBlasEnum(const std::string& option_name, const std::string& x,
                   const std::vector<char>& valid_values) {
  // A valid value contains exactly one character and matches one of the given
  // valid values, ignoring case.
  bool valid = x.size() == 1 && (valid_values.end() !=
                                 std::find_if(valid_values.begin(), valid_values.end(), [&](char v) {
                                   return std::toupper(v) == std::toupper(x[0]);
                                 }));
  if (!valid) {
    std::cout << "Invalid option for --" << option_name << ". Got '" << x << "' but expected one of ";
    for (std::size_t i = 0; i < valid_values.size(); ++i) {
      std::cout << "'" << valid_values[i] << "'";
      if (i != valid_values.size() - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "." << std::endl;
    std::terminate();
  }

  return static_cast<T>(std::toupper(x[0]));
}
}

inline blas::Layout parseLayout(const std::string& layout) {
  return internal::stringToBlasEnum<blas::Layout>("layout", layout, {'C', 'R'});
}

inline blas::Op parseOp(const std::string& op) {
  return internal::stringToBlasEnum<blas::Op>("op", op, {'N', 'T', 'C'});
}

inline blas::Uplo parseUplo(const std::string& uplo) {
  return internal::stringToBlasEnum<blas::Uplo>("uplo", uplo, {'L', 'U', 'G'});
}

inline blas::Diag parseDiag(const std::string& diag) {
  return internal::stringToBlasEnum<blas::Diag>("diag", diag, {'N', 'U'});
}

inline blas::Side parseSide(const std::string& side) {
  return internal::stringToBlasEnum<blas::Side>("side", side, {'L', 'R'});
}

template <SupportReal support_r, SupportComplex support_c>
struct MiniappOptions {
  static constexpr SupportReal support_real = support_r;
  static constexpr SupportComplex support_complex = support_c;

  Backend backend;
  ElementType type;
  int grid_rows;
  int grid_cols;
  int64_t nruns;
  int64_t nwarmups;
  CheckIterFreq do_check;

  MiniappOptions(const hpx::program_options::variables_map& vm)
      : backend(parseBackend(vm["backend"].as<std::string>())),
        type(parseElementType<support_real, support_complex>(vm["type"].as<std::string>())),
        grid_rows(vm["grid-rows"].as<int>()), grid_cols(vm["grid-cols"].as<int>()),
        nruns(vm["nruns"].as<int64_t>()), nwarmups(vm["nwarmups"].as<int64_t>()),
        do_check(parseCheckIterFreq(vm["check-result"].as<std::string>())) {
    DLAF_ASSERT(grid_rows > 0, grid_rows);
    DLAF_ASSERT(grid_cols > 0, grid_cols);
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

  desc.add_options()(
      "backend", hpx::program_options::value<std::string>()->default_value("default"),
      "Backend to use ('default' ('gpu' if available, otherwise 'mc'), 'mc', 'gpu' (if available))");
  desc.add_options()(
      "type", hpx::program_options::value<std::string>()->default_value("d"),
      "Element type to use ('s' (float), 'd' (double), 'c' (std::complex<single>), 'z' (std::complex<double>))");
  desc.add_options()("grid-rows", hpx::program_options::value<int>()->default_value(1),
                     "Number of row processes in the 2D communicator");
  desc.add_options()("grid-cols", hpx::program_options::value<int>()->default_value(1),
                     "Number of column processes in the 2D communicator");
  desc.add_options()("nruns", hpx::program_options::value<int64_t>()->default_value(1),
                     "Number of runs");
  desc.add_options()("nwarmups", hpx::program_options::value<int64_t>()->default_value(1),
                     "Number of warmup runs");
  desc.add_options()("check-result", hpx::program_options::value<std::string>()->default_value("none"),
                     "Enable result checking ('none', 'all', 'last')");

  return desc;
}

inline void addLayoutOption(hpx::program_options::options_description& desc,
                            const blas::Layout def = blas::Layout::ColMajor) {
  desc.add_options()("layout",
                     hpx::program_options::value<std::string>()->default_value({blas::layout2char(def)}),
                     "'C' (ColMajor), 'R' (RowMajor)");
}

inline void addOpOption(hpx::program_options::options_description& desc,
                        const blas::Op def = blas::Op::NoTrans) {
  desc.add_options()("op",
                     hpx::program_options::value<std::string>()->default_value({blas::op2char(def)}),
                     "'N' (NoTrans), 'T' (Trans), 'C' (ConjTrans)");
}

inline void addUploOption(hpx::program_options::options_description& desc,
                          const blas::Uplo def = blas::Uplo::Lower) {
  desc.add_options()("uplo",
                     hpx::program_options::value<std::string>()->default_value({blas::uplo2char(def)}),
                     "'L' (Lower), 'U' (Upper), 'G' (General)");
}

inline void addDiagOption(hpx::program_options::options_description& desc,
                          const blas::Diag def = blas::Diag::NonUnit) {
  desc.add_options()("diag",
                     hpx::program_options::value<std::string>()->default_value({blas::diag2char(def)}),
                     "'N' (NonUnit), 'U' (Unit)");
}

inline void addSideOption(hpx::program_options::options_description& desc,
                          const blas::Side def = blas::Side::Left) {
  desc.add_options()("side",
                     hpx::program_options::value<std::string>()->default_value({blas::side2char(def)}),
                     "'L' (Left), 'R' (Right)");
}
}
