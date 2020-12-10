from itertools import product
from os import system, makedirs
from os.path import expanduser
from re import sub

# Finds two factors of `n` that are as close to each other as possible.
#
# Note: the second factor is larger or equal to the first factor
def sq_factor(n):
    for i in range(1, int(n ** 0.5) + 1):
        if n % i == 0:
            f = (i, n // i)
    return f


# --------------


def init_job_text(run_name, build_dir, nodes, time_min):
    job_text = f"""
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
source {build_dir}/.envrc
export CRAYPE_LINK_TYPE=dynamic
export MPICH_MAX_THREAD_SAFETY=multiple

# Debug
module list &> modules.txt
printenv > env.txt
"""
    return job_text[1:-1]


def submit_job(run_dir, nodes, job_text):
    home_dir = expanduser("~")
    job_path = f"{run_dir}/{nodes}"
    makedirs(job_path, exist_ok=False)
    job_file = f"{job_path}/job.sh"
    with open(job_file, "w") as f:
        f.write(job_text)

    print(f"Submitting : {job_file}")
    system(f"sbatch --chdir={job_path} {job_file}")


# ---------- MINIAPPS ---------------

def chol_ldd(build_dir):
    return f"\n\nldd {build_dir}/miniapp/miniapp_cholesky >> libs.txt"

def chol(
    build_dir,
    nodes,
    rpn,
    nruns,
    matrix_size_arr,
    block_size_arr,
    queue,
    mech,
    pool,
    suffix
):
    if not (rpn == 1 or rpn == 2):
        raise ValueError(f"Wrong value rpn = {rpn}!")

    if queue == "shared":
        queue_flag = "--hpx:queuing=shared-priority"
    elif queue == "default":
        queue_flag = ""
    else:
        raise ValueError(f"Wrong value: queue = {queue}!")

    if mech == "polling":
        mech_flag = "--polling "
    elif mech == "yielding" or mech == "na":
        mech_flag = ""
    else:
        raise ValueError(f"Wrong value: mech = {mech}!")

    if pool == "mpi":
        pool_flags = "--mpipool --hpx:ini=hpx.max_idle_backoff_time=0"
    elif pool == "default" or pool == "na":
        pool_flags = ""
    else:
        raise ValueError(f"Wrong value: pool = {pool}!")


    exe_file = f"{build_dir}/miniapp/miniapp_cholesky"
    cmds = "\n"
    for matrix_size, block_size in product(
        matrix_size_arr,
        block_size_arr,
    ):
        total_ranks = nodes * rpn
        cpus_per_rank = 72 // rpn
        grid_cols, grid_rows = sq_factor(total_ranks)
        cmds += (
            "\nsrun "
            f"-n {total_ranks} "
            f"-c {cpus_per_rank} "
            f"{exe_file} "
            f"--matrix-size {matrix_size} "
            f"--block-size {block_size} "
            f"--grid-rows {grid_rows} "
            f"--grid-cols {grid_cols} "
            f"--nruns {nruns} "
            "--hpx:use-process-mask "
            f"{queue_flag} "
            f"{mech_flag} "
            f"{pool_flags} "
            f">> chol_{suffix}.txt"
        )


    return sub(" +", " ", cmds)


def trsm(
    build_dir,
    nodes,
    ranks_per_node_arr,
    nruns,
    m_arr,
    b_arr,
    extra_flags,
    suffix,
):
    exe_file = "{build_dir}/miniapp/miniapp_triangular_solver"
    cmds = "\n\nldd {exe_file} >> libs.txt"

    n_arr = [x // 2 for x in m_arr]
    for ranks_per_node, m, n, b in product(
        ranks_per_node_arr, m_arr, n_arr, b_arr
    ):
        total_ranks = nodes * ranks_per_node
        cpus_per_rank = 36 * (1 if ranks_per_node == 2 else 2)
        gr, gc = sq_factor(ranks_per_node * nodes)
        cmds += (
            "\nsrun "
            f"-n {total_ranks} "
            f"-c {cpus_per_rank} "
            f"{exe_file} "
            f"--m {m} "
            f"--n {n} "
            f"--mb {b} "
            f"--nb {b} "
            f"--grid-rows {gr} "
            f"--grid-cols {gc} "
            f"--nruns {nruns} "
            "--hpx:use-process-mask "
            f"{extra_flags} "
            f">> trsm_{suffix}.txt"
        )

    return cmds
