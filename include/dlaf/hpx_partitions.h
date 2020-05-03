
#pragma once

/// @file

#include <mpi.h>

#include <hpx/include/resource_partitioner.hpp>
#include <hpx/program_options/options_description.hpp>

namespace dlaf {

// Note: A thread pool must be declared before starting the runtime (hpx::init())
void try_init_mpi_pool(hpx::program_options::options_description& desc, int argc, char** argv) {
  // Declare options before creating resource partitioner
  namespace po = hpx::program_options;
  using hpx::program_options::bool_switch;

  desc.add_options()("mpipool", bool_switch()->default_value(false),
                     "Dedicate a core to MPI if available.");

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).allow_unregistered().options(desc).run(), vm);

  bool use_mpi_pool = vm["mpipool"].as<bool>();

  if (use_mpi_pool) {
    // Create a thread pool for MPI work and add (enabled) PUs on the first core
    //
    hpx::resource::partitioner rp(desc, argc, argv);
    if (rp.numa_domains()[0].cores().size() != 1) {  // if more than 1 core
      rp.create_thread_pool("mpi");
      rp.add_resource(rp.numa_domains()[0].cores()[0].pus(), "mpi");
    }
  }
}

}
