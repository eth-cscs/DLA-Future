#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2022, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

# a system is a dict containing the following data:
# "Cores" (int > 0),
# "Threads per core" (int > 0),
# "Allowed rpns" (list of int > 0),
# "Multiple rpn in same job" (bool),
# "GPU" (bool),
# [optional] "sleep" (int representing the sleep time between runs)
# "Run command" (string:
#                        {nodes}, {rpn}, {total_ranks}, {cores_per_rank}, {threads_per_rank}
#                        will be replaced with the correct value (*).
#                        Extra keywords can be used when providing "Extra subs".)
# "Batch preamble" (multiline string:
#                                     {run_name}, {time_min}, {bs_name}, {nodes}
#                                     will be replaced with the correct value (*),
#                                     if "Multiple rpn in same job" is false
#                                     {rpn}, {total_ranks}, {cores_per_rank}, {threads_per_rank} are also replaced.
#                                     extra keywords can be used when providing "Extra subs".)
# [optional] "Extra subs" (function(dictionary params) which returns a dictionary containing at least the entries of params.
#                          Note: this function is invoked for both "Run command" and "Batch preamble", therefore some items are not always present.)
# (*) Note: replace are done with the format command (extra care needed when using { or }).

cscs = {}

cscs["daint-mc"] = {
    "Cores": 36,
    "Threads per core": 2,
    "Allowed rpns": [1, 2],
    "Multiple rpn in same job": True,
    "GPU": False,
    "Run command": "srun -u -n {total_ranks} -c {threads_per_rank}",
    "Batch preamble": """
#!/bin/bash -l
#SBATCH --job-name={run_name}_{nodes}
#SBATCH --time={time_min}
#SBATCH --nodes={nodes}
#SBATCH --partition=normal
#SBATCH --account=csstaff
#SBATCH --constraint=mc
#SBATCH --output=output.txt
#SBATCH --error=error.txt

# Env
export MPICH_MAX_THREAD_SAFETY=multiple

# Debug
module list &> modules_{bs_name}.txt
printenv > env_{bs_name}.txt

# Commands
""",
}

cscs["daint-gpu"] = {
    "Cores": 12,
    "Threads per core": 2,
    "Allowed rpns": [1],
    "Multiple rpn in same job": True,
    "GPU": True,
    "Run command": "srun -u -n {total_ranks} -c {threads_per_rank}",
    "Batch preamble": """
#!/bin/bash -l
#SBATCH --job-name={run_name}_{nodes}
#SBATCH --time={time_min}
#SBATCH --nodes={nodes}
#SBATCH --partition=normal
#SBATCH --account=csstaff
#SBATCH --constraint=gpu
#SBATCH --output=output.txt
#SBATCH --error=error.txt

# Env
export MPICH_MAX_THREAD_SAFETY=multiple

# Debug
module list &> modules_{bs_name}.txt
printenv > env_{bs_name}.txt

# Commands
""",
}

cscs["eiger"] = {
    "Cores": 128,
    "Threads per core": 2,
    "Allowed rpns": [1, 2, 4, 8],
    "Multiple rpn in same job": True,
    "GPU": False,
    "Run command": "srun -u -n {total_ranks} --cpu-bind=core -c {threads_per_rank}",
    "Batch preamble": """
#!/bin/bash -l
#SBATCH --job-name={run_name}_{nodes}
#SBATCH --time={time_min}
#SBATCH --nodes={nodes}
#SBATCH --account=csstaff
#SBATCH --output=output.txt
#SBATCH --error=error.txt

# Env
export MPICH_MAX_THREAD_SAFETY=multiple

# Debug
module list &> modules_{bs_name}.txt
printenv > env_{bs_name}.txt

# Commands
""",
}

cineca = {}


def extraSubsMarconi(params):
    if params["nodes"] <= 16:
        params["qos"] = "normal"
    else:
        params["qos"] = "m100_qos_bprod"

    # This configuration was suggested by the user support and used for the benchmarks.
    # It looks strange that the computation should be constrained to only half the cores in the node.
    # Extra investigations are needed.
    if "rpn" in params:
        if params["rpn"] == 1:
            params["socket_PE"] = 16
        else:
            params["socket_PE"] = 8
    return params


cineca["m100_cpu"] = {
    "Cores": 32,
    "Threads per core": 4,
    "Allowed rpns": [2, 4],
    "Multiple rpn in same job": False,
    "GPU": False,
    "sleep": 5,
    "Run command": "mpirun --rank-by core --map-by socket:PE={socket_PE}",
    "Batch preamble": """
#!/bin/bash -l
#SBATCH --job-name={run_name}_{nodes}
#SBATCH --time={time_min}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={rpn}
#SBATCH --cpus-per-task={threads_per_rank}
#SBATCH --partition=m100_usr_prod
#SBATCH --account=cin_staff
#SBATCH --qos={qos}
#SBATCH --output=output.txt
#SBATCH --error=error.txt

# Debug
module list &> modules_{bs_name}.txt
printenv > env_{bs_name}.txt

# Commands
""",
    "Extra subs": extraSubsMarconi,
}

# NOTE: Here is assumed that `gpu2ranks_ompi` is in PATH!
#       modify "Run command" if it is not the case.
cineca["m100"] = {
    "Cores": 32,
    "Threads per core": 4,
    "Allowed rpns": [2, 4],
    "Multiple rpn in same job": False,
    "GPU": True,
    "sleep": 5,
    "Run command": "mpirun --rank-by core --map-by socket:PE={socket_PE} gpu2ranks_ompi",
    "Batch preamble": """
#!/bin/bash -l
#SBATCH --job-name={run_name}_{nodes}
#SBATCH --time={time_min}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={rpn}
#SBATCH --cpus-per-task={threads_per_rank}
#SBATCH --partition=m100_usr_prod
#SBATCH --gres=gpu:{rpn}
#SBATCH --account=cin_staff
#SBATCH --qos={qos}
#SBATCH --output=output.txt
#SBATCH --error=error.txt

# Debug
module list &> modules_{bs_name}.txt
printenv > env_{bs_name}.txt

# NOTE: It is assumed that `gpu2ranks_ompi` is in PATH!
#       modify "Run command" in `systems.py` if it is not the case.

# Commands
""",
    "Extra subs": extraSubsMarconi,
}
