//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2020-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/init.h"

#include <cstdlib>

#include <gtest/gtest.h>

#include <hpx/init.hpp>

static const char* binary_name = "init_test";
static const char* env_var_name = "DLAF_NUM_HP_CUDA_STREAMS_PER_THREAD";
static const char* command_line_option_name = "--dlaf:num-hp-cuda-streams-per-thread";

static int argc_without_option = 1;
static const char* argv_without_option[] = {binary_name};

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
    dlaf::initialize(argc_without_option, argv_without_option);
    dlaf::configuration cfg = dlaf::internal::getConfiguration();
    EXPECT_EQ(default_val, cfg.num_hp_cuda_streams_per_thread);
    dlaf::finalize();
  }

  // User configuration should take precedence over default configuration.
  {
    dlaf::configuration user_cfg = default_cfg;
    user_cfg.num_hp_cuda_streams_per_thread = default_val + 1;

    dlaf::initialize(argc_without_option, argv_without_option, user_cfg);
    dlaf::configuration cfg = dlaf::internal::getConfiguration();
    EXPECT_EQ(user_cfg.num_hp_cuda_streams_per_thread, cfg.num_hp_cuda_streams_per_thread);
    dlaf::finalize();
  }

  // Environment variables should take precedence over user configuration.
  {
    dlaf::configuration user_cfg = default_cfg;
    user_cfg.num_hp_cuda_streams_per_thread = default_val + 1;
    const std::size_t env_var_val = user_cfg.num_hp_cuda_streams_per_thread + 1;
    const std::string env_var_val_str = std::to_string(env_var_val);
    setenv(env_var_name, env_var_val_str.c_str(), 1);

    dlaf::initialize(argc_without_option, argv_without_option, user_cfg);
    dlaf::configuration cfg = dlaf::internal::getConfiguration();
    EXPECT_EQ(env_var_val, cfg.num_hp_cuda_streams_per_thread);
    dlaf::finalize();
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

    dlaf::initialize(argc_with_option, argv_with_option, user_cfg);
    dlaf::configuration cfg = dlaf::internal::getConfiguration();
    EXPECT_EQ(command_line_option_val, cfg.num_hp_cuda_streams_per_thread);
    dlaf::finalize();
  }

  return hpx::finalize();
}

TEST(Init, Precedence) {
  // The const_cast is currently necessary for hpx::init. HPX should be updated
  // to take const argc/argv.
  hpx::init(precedence_main, argc_without_option, const_cast<char**>(argv_without_option));
}

int vm_no_command_line_option_main(hpx::program_options::variables_map& vm) {
  const dlaf::configuration default_cfg;
  const std::size_t default_val = default_cfg.num_hp_cuda_streams_per_thread;
  // Note that this test doesn't mean that the default value has to be 3. It is
  // included to help catch unexpected changes in the configuration handling.
  EXPECT_EQ(default_val, 3);

  // Make sure environment is clean for the test.
  unsetenv(env_var_name);

  // Default configuration.
  {
    dlaf::initialize(vm);
    dlaf::configuration cfg = dlaf::internal::getConfiguration();
    EXPECT_EQ(default_val, cfg.num_hp_cuda_streams_per_thread);
    dlaf::finalize();
  }

  // User configuration should take precedence over default configuration.
  {
    dlaf::configuration user_cfg = default_cfg;
    user_cfg.num_hp_cuda_streams_per_thread = default_val + 1;

    dlaf::initialize(vm, user_cfg);
    dlaf::configuration cfg = dlaf::internal::getConfiguration();
    EXPECT_EQ(user_cfg.num_hp_cuda_streams_per_thread, cfg.num_hp_cuda_streams_per_thread);
    dlaf::finalize();
  }

  // Environment variables should take precedence over user configuration.
  {
    dlaf::configuration user_cfg = default_cfg;
    user_cfg.num_hp_cuda_streams_per_thread = default_val + 1;
    const std::size_t env_var_val = user_cfg.num_hp_cuda_streams_per_thread + 1;
    const std::string env_var_val_str = std::to_string(env_var_val);
    setenv(env_var_name, env_var_val_str.c_str(), 1);

    dlaf::initialize(vm, user_cfg);
    dlaf::configuration cfg = dlaf::internal::getConfiguration();
    EXPECT_EQ(env_var_val, cfg.num_hp_cuda_streams_per_thread);
    dlaf::finalize();
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

    dlaf::initialize(argc_with_option, argv_with_option, user_cfg);
    dlaf::configuration cfg = dlaf::internal::getConfiguration();
    EXPECT_EQ(command_line_option_val, cfg.num_hp_cuda_streams_per_thread);
    dlaf::finalize();
  }

  return hpx::finalize();
}

TEST(Init, VariablesMapNoCommandLineOption) {
  // The const_cast is currently necessary for hpx::init. HPX should be updated
  // to take const argc/argv.
  hpx::init(vm_no_command_line_option_main, argc_without_option,
            const_cast<char**>(argv_without_option));
}

int vm_command_line_option_main(hpx::program_options::variables_map& vm) {
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

    dlaf::initialize(vm, user_cfg);
    dlaf::configuration cfg = dlaf::internal::getConfiguration();
    EXPECT_EQ(command_line_option_val, cfg.num_hp_cuda_streams_per_thread);
    dlaf::finalize();
  }

  return hpx::finalize();
}

TEST(Init, VariablesMapCommandLineOption) {
  hpx::program_options::options_description options("options");
  options.add(dlaf::getOptionsDescription());

  hpx::init_params p;
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
  // The const_cast is currently necessary for hpx::init. HPX should be updated
  // to take const argc/argv.
  char* argv_with_option[] = {const_cast<char*>(binary_name),
                              const_cast<char*>(command_line_option_str.c_str())};

  hpx::init(vm_command_line_option_main, argc_with_option, argv_with_option, p);
}
