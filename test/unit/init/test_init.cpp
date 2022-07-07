//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/init.h"

#include <cstdlib>

#include <gtest/gtest.h>

#include <pika/init.hpp>

static const char* binary_name = "init_test";
static const char* env_var_name = "DLAF_NUM_HP_CUDA_STREAMS_PER_THREAD";
static const char* command_line_option_name = "--dlaf:num-hp-cuda-streams-per-thread";

static int argc_without_option = 1;
static const char* argv_without_option[] = {binary_name};

enum class InitializerType {
  RAII,
  InitializeFinalize,
};

// This helper primarily exists to test that the free function dlaf::finalize
// and the RAII helper dlaf::ScopedInitializer constructor accept the same
// arguments and behave the same.
struct InitializeTester {
  InitializerType type_;
  std::unique_ptr<dlaf::ScopedInitializer> init_ = nullptr;

  template <typename... Args>
  InitializeTester(InitializerType type, Args&&... args) : type_(type) {
    if (type_ == InitializerType::RAII) {
      init_ = std::make_unique<dlaf::ScopedInitializer>(std::forward<Args>(args)...);
    }
    else if (type_ == InitializerType::InitializeFinalize) {
      dlaf::initialize(std::forward<Args>(args)...);
    }
    else {
      EXPECT_TRUE(false);
    }
  }

  ~InitializeTester() {
    if (type_ == InitializerType::RAII) {
    }
    else if (type_ == InitializerType::InitializeFinalize) {
      dlaf::finalize();
    }
    else {
      EXPECT_TRUE(false);
    }
  }

  InitializeTester(InitializeTester&&) = delete;
  InitializeTester(InitializeTester const&) = delete;
  InitializeTester& operator=(InitializeTester&&) = delete;
  InitializeTester& operator=(InitializeTester const&) = delete;
};

static InitializerType current_initializer_type;

template <typename F>
void initialize_tester_helper(F&& f) {
  current_initializer_type = InitializerType::RAII;
  f();

  current_initializer_type = InitializerType::InitializeFinalize;
  f();
}

const auto initializer_types =
    ::testing::Values(InitializerType::RAII, InitializerType::InitializeFinalize);

class InitTest : public ::testing::TestWithParam<InitializerType> {};

int precedence_main(int, char*[]) {
  const dlaf::configuration default_cfg;
  const std::size_t default_val = default_cfg.num_hp_cuda_streams_per_thread;
  // Note that this test doesn't mean that the default value has to be 3. It is
  // included to help catch unexpected changes in the configuration handling.
  EXPECT_EQ(default_val, 3);

  // Make sure environment is clean for the test.
  unsetenv(env_var_name);

  // Default configuration.
  {
    InitializeTester init(current_initializer_type, argc_without_option, argv_without_option);
    dlaf::configuration cfg = dlaf::internal::getConfiguration();
    EXPECT_EQ(default_val, cfg.num_hp_cuda_streams_per_thread);
  }

  // User configuration should take precedence over default configuration.
  {
    dlaf::configuration user_cfg = default_cfg;
    user_cfg.num_hp_cuda_streams_per_thread = default_val + 1;

    InitializeTester init(current_initializer_type, argc_without_option, argv_without_option, user_cfg);
    dlaf::configuration cfg = dlaf::internal::getConfiguration();
    EXPECT_EQ(user_cfg.num_hp_cuda_streams_per_thread, cfg.num_hp_cuda_streams_per_thread);
  }

  // Environment variables should take precedence over user configuration.
  {
    dlaf::configuration user_cfg = default_cfg;
    user_cfg.num_hp_cuda_streams_per_thread = default_val + 1;
    const std::size_t env_var_val = user_cfg.num_hp_cuda_streams_per_thread + 1;
    const std::string env_var_val_str = std::to_string(env_var_val);
    setenv(env_var_name, env_var_val_str.c_str(), 1);

    InitializeTester init(current_initializer_type, argc_without_option, argv_without_option, user_cfg);
    dlaf::configuration cfg = dlaf::internal::getConfiguration();
    EXPECT_EQ(env_var_val, cfg.num_hp_cuda_streams_per_thread);
  }

  // Command-line options should take precedence over environment variables.
  {
    dlaf::configuration user_cfg = default_cfg;
    user_cfg.num_hp_cuda_streams_per_thread = default_val + 1;
    const std::size_t env_var_val = user_cfg.num_hp_cuda_streams_per_thread + 1;
    const std::string env_var_val_str = std::to_string(env_var_val);
    setenv(env_var_name, env_var_val_str.c_str(), 1);
    const std::size_t command_line_option_val = env_var_val + 1;
    const std::string command_line_option_val_str = std::to_string(command_line_option_val);
    const std::string command_line_option_str =
        command_line_option_name + std::string("=") + command_line_option_val_str;
    const int argc_with_option = 2;
    const char* argv_with_option[] = {binary_name, command_line_option_str.c_str()};

    InitializeTester init(current_initializer_type, argc_with_option, argv_with_option, user_cfg);
    dlaf::configuration cfg = dlaf::internal::getConfiguration();
    EXPECT_EQ(command_line_option_val, cfg.num_hp_cuda_streams_per_thread);
  }

  return pika::finalize();
}

TEST_P(InitTest, Precedence) {
  current_initializer_type = GetParam();

  pika::init(precedence_main, argc_without_option, argv_without_option);
}

int vm_no_command_line_option_main(pika::program_options::variables_map& vm) {
  const dlaf::configuration default_cfg;
  const std::size_t default_val = default_cfg.num_hp_cuda_streams_per_thread;
  // Note that this test doesn't mean that the default value has to be 3. It is
  // included to help catch unexpected changes in the configuration handling.
  EXPECT_EQ(default_val, 3);

  // Make sure environment is clean for the test.
  unsetenv(env_var_name);

  // Default configuration.
  {
    InitializeTester init(current_initializer_type, vm);
    dlaf::configuration cfg = dlaf::internal::getConfiguration();
    EXPECT_EQ(default_val, cfg.num_hp_cuda_streams_per_thread);
  }

  // User configuration should take precedence over default configuration.
  {
    dlaf::configuration user_cfg = default_cfg;
    user_cfg.num_hp_cuda_streams_per_thread = default_val + 1;

    InitializeTester init(current_initializer_type, vm, user_cfg);
    dlaf::configuration cfg = dlaf::internal::getConfiguration();
    EXPECT_EQ(user_cfg.num_hp_cuda_streams_per_thread, cfg.num_hp_cuda_streams_per_thread);
  }

  // Environment variables should take precedence over user configuration.
  {
    dlaf::configuration user_cfg = default_cfg;
    user_cfg.num_hp_cuda_streams_per_thread = default_val + 1;
    const std::size_t env_var_val = user_cfg.num_hp_cuda_streams_per_thread + 1;
    const std::string env_var_val_str = std::to_string(env_var_val);
    setenv(env_var_name, env_var_val_str.c_str(), 1);

    InitializeTester init(current_initializer_type, vm, user_cfg);
    dlaf::configuration cfg = dlaf::internal::getConfiguration();
    EXPECT_EQ(env_var_val, cfg.num_hp_cuda_streams_per_thread);
  }

  // Command-line options should take precedence over environment variables.
  {
    dlaf::configuration user_cfg = default_cfg;
    user_cfg.num_hp_cuda_streams_per_thread = default_val + 1;
    const std::size_t env_var_val = user_cfg.num_hp_cuda_streams_per_thread + 1;
    const std::string env_var_val_str = std::to_string(env_var_val);
    setenv(env_var_name, env_var_val_str.c_str(), 1);
    const std::size_t command_line_option_val = env_var_val + 1;
    const std::string command_line_option_val_str = std::to_string(command_line_option_val);
    const std::string command_line_option_str =
        command_line_option_name + std::string("=") + command_line_option_val_str;
    const int argc_with_option = 2;
    const char* argv_with_option[] = {binary_name, command_line_option_str.c_str()};

    InitializeTester init(current_initializer_type, argc_with_option, argv_with_option, user_cfg);
    dlaf::configuration cfg = dlaf::internal::getConfiguration();
    EXPECT_EQ(command_line_option_val, cfg.num_hp_cuda_streams_per_thread);
  }

  return pika::finalize();
}

TEST_P(InitTest, VariablesMapNoCommandLineOption) {
  current_initializer_type = GetParam();

  pika::init(vm_no_command_line_option_main, argc_without_option, argv_without_option);
}

int vm_command_line_option_main(pika::program_options::variables_map& vm) {
  dlaf::configuration default_cfg;
  const std::size_t default_val = default_cfg.num_hp_cuda_streams_per_thread;

  // Command-line options should take precedence over everything else.
  {
    dlaf::configuration user_cfg = default_cfg;
    user_cfg.num_hp_cuda_streams_per_thread = default_val + 1;
    const std::size_t env_var_val = user_cfg.num_hp_cuda_streams_per_thread + 1;
    const std::string env_var_val_str = std::to_string(env_var_val);
    setenv(env_var_name, env_var_val_str.c_str(), 1);
    const std::size_t command_line_option_val = env_var_val + 1;

    InitializeTester init(current_initializer_type, vm, user_cfg);
    dlaf::configuration cfg = dlaf::internal::getConfiguration();
    EXPECT_EQ(command_line_option_val, cfg.num_hp_cuda_streams_per_thread);
  }

  return pika::finalize();
}

TEST_P(InitTest, VariablesMapCommandLineOption) {
  current_initializer_type = GetParam();

  pika::program_options::options_description options("options");
  options.add(dlaf::getOptionsDescription());

  pika::init_params p;
  p.desc_cmdline = options;

  dlaf::configuration default_cfg;
  const std::size_t default_val = default_cfg.num_hp_cuda_streams_per_thread;
  // Note that this test doesn't mean that the default value has to be 3. It is
  // included to help catch unexpected changes in the configuration handling.
  EXPECT_EQ(default_val, 3);

  std::size_t command_line_option_val = default_val + 3;
  std::string command_line_option_val_str = std::to_string(command_line_option_val);
  std::string command_line_option_str =
      command_line_option_name + std::string("=") + command_line_option_val_str;
  int argc_with_option = 2;
  const char* argv_with_option[] = {binary_name, command_line_option_str.c_str()};

  pika::init(vm_command_line_option_main, argc_with_option, argv_with_option, p);
}

INSTANTIATE_TEST_SUITE_P(Init, InitTest, initializer_types);
