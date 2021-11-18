cscs = {}

cscs["daint-mc"] = {
    "Cores": 36,
    "Threads per core": 2,
    "Allowed rpns": [1, 2],
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
module list &> modules.txt
printenv > env.txt

# Commands
""",
}

cscs["daint-gpu"] = {
    "Cores": 12,
    "Threads per core": 2,
    "Allowed rpns": [1],
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
module list &> modules.txt
printenv > env.txt

# Commands
""",
}

cscs["eiger"] = {
    "Cores": 128,
    "Threads per core": 2,
    "Allowed rpns": [1, 2, 4, 8],
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
module list &> modules.txt
printenv > env.txt

# Commands
""",
}

# NOTE: `gpu2ranks_ompi` has to be in PATH! (e.g. in `~/bin`)
cineca["m100-gpu"] = {
    "Cores": 16,
    "Threads per core": 2,
    "Allowed rpns": [2],  # 4?
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
