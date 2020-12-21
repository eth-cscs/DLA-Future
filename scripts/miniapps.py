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


def chol_ldd(build_dir):
    return f"\n\nldd {build_dir}/miniapp/miniapp_cholesky >> libs.txt"


def trsm_ldd(build_dir):
    return f"\n\nldd {build_dir}/miniapp/miniapp_triangular_solver >> libs.txt"


def _get_queue_flag(queue):
    if queue == "shared":
        return "--hpx:queuing=shared-priority"
    elif queue == "default":
        return ""
    else:
        raise ValueError(f"Wrong value: queue = {queue}!")


def _get_mech_flag(mech):
    if mech == "polling":
        return "--polling "
    elif mech == "yielding" or mech == "na":
        return ""
    else:
        raise ValueError(f"Wrong value: mech = {mech}!")


def _get_pool_flag(pool):
    if pool == "mpi":
        pool_flags = "--mpipool --hpx:ini=hpx.max_idle_backoff_time=0"
    elif pool == "default" or pool == "na":
        pool_flags = ""
    else:
        raise ValueError(f"Wrong value: pool = {pool}!")


def _check_ranks_per_node(rpn):
    if not (rpn == 1 or rpn == 2):
        raise ValueError(f"Wrong value rpn = {rpn}!")


# ---------- MINIAPPS ---------------


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
    suffix,
):
    _check_ranks_per_node(rpn)
    queue_flag = _get_queue_flag(queue)
    mech_flag = _get_mech_flag(mech)
    pool_flag = _get_pool_flag(pool)
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
            f"{build_dir}/miniapp/miniapp_cholesky "
            f"--matrix-size {matrix_size} "
            f"--block-size {block_size} "
            f"--grid-rows {grid_rows} "
            f"--grid-cols {grid_cols} "
            f"--nruns {nruns} "
            "--hpx:use-process-mask "
            f"{queue_flag} "
            f"{mech_flag} "
            f"{pool_flags} "
            f">> chol_{suffix}.out"
        )

    return sub(" +", " ", cmds)


def trsm(
    build_dir,
    nodes,
    rpn,
    nruns,
    m_arr,
    b_arr,
    suffix,
):
    _check_ranks_per_node(rpn)
    n_arr = [x // 2 for x in m_arr]
    cmds = "\n"
    for m, n, b in product(m_arr, n_arr, b_arr):
        total_ranks = nodes * rpn
        cpus_per_rank = 72 // rpn
        gr, gc = sq_factor(total_ranks)
        cmds += (
            "\nsrun "
            f"-n {total_ranks} "
            f"-c {cpus_per_rank} "
            f"{build_dir}/miniapp/miniapp_triangular_solver "
            f"--m {m} "
            f"--n {n} "
            f"--mb {b} "
            f"--nb {b} "
            f"--grid-rows {gr} "
            f"--grid-cols {gc} "
            f"--nruns {nruns} "
            "--hpx:use-process-mask "
            f">> trsm_{suffix}.out"
        )

    return sub(" +", " ", cmds)
