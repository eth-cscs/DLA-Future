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

#include <algorithm>
#include <cctype>
#include <exception>
#include <iostream>
#include <string>
#include <type_traits>

#include <blas.hh>

#include <pika/program_options.hpp>

#include <dlaf/common/assert.h>
#include <dlaf/common/format_short.h>
#include <dlaf/types.h>

#define DLAF_MINIAPP_UNSUPPORTED_OPTION_VALUE(option, actual)                             \
  std::cout << "Valid but unsupported option for " << option << ": '" << actual << "\'."; \
  std::terminate();

#define DLAF_MINIAPP_INVALID_OPTION_VALUE(option, actual, expected)                                 \
  std::cout << "Invalid option for " << option << ". Got \'" << actual << "\' but expected one of " \
            << expected << ".";                                                                     \
  std::terminate();

namespace dlaf::miniapp {
inline Backend parseBackend(const std::string& backend) {
  if (backend == "default")
    return Backend::Default;
  else if (backend == "mc")
    return Backend::MC;
  else if (backend == "gpu") {
#if !defined(DLAF_WITH_GPU)
    std::cout << "Asked for --backend=gpu but both DLAF_WITH_CUDA and DLAF_WITH_HIP are disabled!" << std::endl;
    std::terminate();
#endif
    return Backend::GPU;
  }

  DLAF_MINIAPP_INVALID_OPTION_VALUE("--backend", backend, "'default', 'mc', 'gpu' (if available)");
  return DLAF_UNREACHABLE(Backend);
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
        DLAF_MINIAPP_UNSUPPORTED_OPTION_VALUE("--type", type);
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
        DLAF_MINIAPP_UNSUPPORTED_OPTION_VALUE("--type", type);
      }
    }
  }

  DLAF_MINIAPP_INVALID_OPTION_VALUE("--type", type, "'s', 'd', 'c', 'z'");
  return DLAF_UNREACHABLE(ElementType);
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

  DLAF_MINIAPP_INVALID_OPTION_VALUE("--check-result", check, "'none', 'last', 'all'");
  return DLAF_UNREACHABLE(CheckIterFreq);
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
    std::ostringstream valid_values_stream;
    for (std::size_t i = 0; i < valid_values.size(); ++i) {
      valid_values_stream << "'" << valid_values[i] << "'";
      if (i != valid_values.size() - 1) {
        valid_values_stream << ", ";
      }
    }
    std::string option_name_dashes = "--" + option_name;
    DLAF_MINIAPP_INVALID_OPTION_VALUE(option_name, x, valid_values_stream.str());
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

  MiniappOptions(const pika::program_options::variables_map& vm)
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

inline pika::program_options::options_description getMiniappOptionsDescription() {
  pika::program_options::options_description desc("DLA-Future miniapp options");

  desc.add_options()(
      "backend", pika::program_options::value<std::string>()->default_value("default"),
      "Backend to use ('default' ('gpu' if available, otherwise 'mc'), 'mc', 'gpu' (if available))");
  desc.add_options()(
      "type", pika::program_options::value<std::string>()->default_value("d"),
      "Element type to use ('s' (float), 'd' (double), 'c' (std::complex<single>), 'z' (std::complex<double>))");
  desc.add_options()("grid-rows", pika::program_options::value<int>()->default_value(1),
                     "Number of row processes in the 2D communicator");
  desc.add_options()("grid-cols", pika::program_options::value<int>()->default_value(1),
                     "Number of column processes in the 2D communicator");
  desc.add_options()("nruns", pika::program_options::value<int64_t>()->default_value(1),
                     "Number of runs");
  desc.add_options()("nwarmups", pika::program_options::value<int64_t>()->default_value(1),
                     "Number of warmup runs");
  desc.add_options()("check-result", pika::program_options::value<std::string>()->default_value("none"),
                     "Enable result checking ('none', 'all', 'last')");

  return desc;
}

template <SupportReal support_r, SupportComplex support_c>
struct MiniappKernelOptions {
  static constexpr SupportReal support_real = support_r;
  static constexpr SupportComplex support_complex = support_c;

  Backend backend;
  ElementType type;
  int64_t nruns;
  int64_t nparallel;
  int64_t count;
  CheckIterFreq do_check;

  MiniappKernelOptions(const pika::program_options::variables_map& vm)
      : backend(parseBackend(vm["backend"].as<std::string>())),
        type(parseElementType<support_real, support_complex>(vm["type"].as<std::string>())),
        nruns(vm["nruns"].as<int64_t>()), nparallel(vm["nparallel"].as<int64_t>()),
        count(vm["count"].as<int64_t>()),
        do_check(parseCheckIterFreq(vm["check-result"].as<std::string>())) {
    DLAF_ASSERT(nruns > 0, nruns);
    DLAF_ASSERT(nparallel > 0, nparallel);
    DLAF_ASSERT(count > 0, count);
  }

  MiniappKernelOptions(MiniappKernelOptions&&) = default;
  MiniappKernelOptions(const MiniappKernelOptions&) = default;
  MiniappKernelOptions& operator=(MiniappKernelOptions&&) = default;
  MiniappKernelOptions& operator=(const MiniappKernelOptions&) = default;
};

inline pika::program_options::options_description getMiniappKernelOptionsDescription() {
  pika::program_options::options_description desc("DLA-Future kernel miniapp options");

  desc.add_options()("help,h", "produce help message");

  desc.add_options()(
      "backend", pika::program_options::value<std::string>()->default_value("default"),
      "Backend to use ('default' ('gpu' if available, otherwise 'mc'), 'mc', 'gpu' (if available))");
  desc.add_options()(
      "type", pika::program_options::value<std::string>()->default_value("d"),
      "Element type to use ('s' (float), 'd' (double), 'c' (std::complex<single>), 'z' (std::complex<double>))");
  desc.add_options()("nruns", pika::program_options::value<int64_t>()->default_value(1),
                     "Number of runs");
  desc.add_options()("nparallel", pika::program_options::value<int64_t>()->default_value(1),
                     "Number of operation allowed in parallel (i.e. CPU threads or CUDA streams used)");
  desc.add_options()("count", pika::program_options::value<int64_t>()->default_value(10),
                     "Total number of operations scheduled");
  desc.add_options()("check-result", pika::program_options::value<std::string>()->default_value("none"),
                     "Enable result checking ('none', 'all', 'last')");

  return desc;
}

inline void addLayoutOption(pika::program_options::options_description& desc,
                            const blas::Layout def = blas::Layout::ColMajor) {
  desc.add_options()("layout",
                     pika::program_options::value<std::string>()->default_value(
                         {blas::layout2char(def)}),
                     "'C' (ColMajor), 'R' (RowMajor)");
}

inline void addOpOption(pika::program_options::options_description& desc,
                        const blas::Op def = blas::Op::NoTrans) {
  desc.add_options()("op",
                     pika::program_options::value<std::string>()->default_value({blas::op2char(def)}),
                     "'N' (NoTrans), 'T' (Trans), 'C' (ConjTrans)");
}

inline void addUploOption(pika::program_options::options_description& desc,
                          const blas::Uplo def = blas::Uplo::Lower) {
  desc.add_options()("uplo",
                     pika::program_options::value<std::string>()->default_value({blas::uplo2char(def)}),
                     "'L' (Lower), 'U' (Upper), 'G' (General)");
}

inline void addDiagOption(pika::program_options::options_description& desc,
                          const blas::Diag def = blas::Diag::NonUnit) {
  desc.add_options()("diag",
                     pika::program_options::value<std::string>()->default_value({blas::diag2char(def)}),
                     "'N' (NonUnit), 'U' (Unit)");
}

inline void addSideOption(pika::program_options::options_description& desc,
                          const blas::Side def = blas::Side::Left) {
  desc.add_options()("side",
                     pika::program_options::value<std::string>()->default_value({blas::side2char(def)}),
                     "'L' (Left), 'R' (Right)");
}
}
