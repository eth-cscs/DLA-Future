# a system is a dict containing the following data:
# "Cores" (int > 0),
# "Threads per core" (int > 0),
# "Allowed rpns" (list of int > 0),
# "Multiple rpn in same job" (bool),
# "GPU" (bool),
# "Run command" (string ({nodes}, {rpn}, {total_ranks}, {cores_per_rank}, {threads_per_rank} will be replaced with the correct value))
# "Batch preamble" (multiline string ({run_name}, {time_min}, {bs_name}, {nodes} will be replaced with the correct value,
#                  if "Multiple rpn in same job" is false {rpn}, {total_ranks}, {cores_per_rank}, {threads_per_rank} are also replaced.)
# Note: replace are done with the format command (extra care needed when using { or }).

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

# NOTE: `gpu2ranks_ompi` has to be in PATH! (e.g. in `~/bin`)
cineca["m100-gpu"] = {
    "Cores": 16,
    "Threads per core": 2,
    "Allowed rpns": [2],  # 4?
    "GPU": True,
    "Run command": "mpirun gpu2ranks_ompi",
    "Batch preamble": """
#!/bin/bash -l
#SBATCH --job-name={run_name}_{nodes}
#SBATCH --time={time_min}
#SBATCH --nodes={nodes}
#SBATCH --partition=m100_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --qos=normal
#SBATCH --output=output.txt
#SBATCH --error=error.txt

module load spectrum_mpi/10.3.1--binary
module load dla-future-develop-gcc-10.2.0-ruvjcr5

# Debug
module list &> modules.txt
printenv > env.txt

# Commands
""",
}
