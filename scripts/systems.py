cscs = {}

cscs["daint-mc"] = {
  "Cores": 36,
  "Threads per core": 2,
  "Allowed rpns": [1, 2],
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
"""
}

cscs["daint-gpu"] = {
  "Cores": 12,
  "Threads per core": 2,
  "Allowed rpns": [1],
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
"""
}

cscs["eiger"] = {
  "Cores": 128,
  "Threads per core": 2,
  "Allowed rpns": [1, 2, 4, 8],
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
"""
}
