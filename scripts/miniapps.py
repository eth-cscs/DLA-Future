from os import system, makedirs
from os.path import expanduser
from re import sub

# Finds two factors of `n` that are as close to each other as possible.
#
# Note: the second factor is larger or equal to the first factor
def _sq_factor(n):
    for i in range(1, int(n ** 0.5) + 1):
        if n % i == 0:
            f = (i, n // i)
    return f


def _check_ranks_per_node(rpn):
    if not (rpn == 1 or rpn == 2):
        raise ValueError(f"Wrong value rpn = {rpn}!")


# parse dlaf_nb_[1|0][1|0][1|0]
#       dlaf_nb_<hpx queue><non-blocking mechanism><mpi pool>
#       dlaf
def _dlaf_flags(lib):
    if lib == "dlaf":
        return ""

    q, p, m = tuple(lib[8:])
    return "{} {} {}".format(
        "--hpx:queuing=shared-priority" if q == "1" else "",
        "--polling" if p == "1" else "",
        "--mpipool --hpx:ini=hpx.max_idle_backoff_time=0" if m == "1" else "",
    )


def _err_msg(lib):
    return f"No such `lib`: {lib}! Allowed values are : `dlaf|dlaf_nb_[0|1][0|1][0|1]`, `slate` and `dplasma`."


# Job preamble
#
def init_job_text(run_name, nodes, time_min):
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
export MPICH_MAX_THREAD_SAFETY=multiple

# Debug
module list &> modules.txt
printenv > env.txt
"""
    return job_text[1:-1]


# Create the job directory tree and submit jobs.
#
def submit_jobs(run_dir, nodes, job_text, suffix = ""):
    job_path = expanduser(f"{run_dir}/{nodes}")
    makedirs(job_path, exist_ok=False)
    job_file = f"{job_path}/job{suffix}.sh"
    with open(job_file, "w") as f:
        f.write(job_text)

    print(f"Submitting : {job_file}")
    system(f"sbatch --chdir={job_path} {job_file}")


# lib: allowed libraries are dlaf|dlaf_nb_[0|1][0|1][0|1]|slate|dplasma
#
# where `dlaf_nb_[0|1][0|1][0|1]` is `dlaf_nb_<hpx queue><non-blocking mechanism><mpi pool>`, for example
#       `dlaf_nb_111` : dlaf with non-blocking MPI(nb), shared priority queue(1), polling (1) and a dedicated core for an MPI pool (1).
#
# rpn: ranks per node
#
def chol(lib, build_dir, nodes, rpn, m_sz, mb_sz, nruns):
    _check_ranks_per_node(rpn)

    total_ranks = nodes * rpn
    cpus_per_rank = 72 // rpn
    grid_cols, grid_rows = _sq_factor(total_ranks)

    if lib.startswith("dlaf"):
        extra_flags = _dlaf_flags(lib)
        cmd = f"{build_dir}/miniapp/miniapp_cholesky --matrix-size {m_sz} --block-size {mb_sz} --grid-rows {grid_rows} --grid-cols {grid_cols} --nruns {nruns} --hpx:use-process-mask {extra_flags}"
    elif lib == "slate":
        cmd = f"{build_dir}/test/slate_test potrf --dim {m_sz}x${m_sz}x0 --nb {mb_sz} --p {grid_rows} --q {grid_cols} --repeat {nruns} --check n --ref n --type d"
    elif lib == "dplasma":
        if rpn != 1:
            ValueError("DPLASMA can only run with 1 rank per node!")
        cmd = f"{build_dir}/tests/testing_dpotrf -N ${m_sz} --MB ${mb_sz} --NB ${mb_sz} --grid-rows ${grid_rows} --grid-cols ${grid_cols} -c 36 -v"
    else:
        raise ValueError(_err_msg(lib))

    return f"\nsrun -n {total_ranks} -c {cpus_per_rank} {cmd} >> chol_{lib}.out"


# lib: allowed libraries are dlaf|dlaf_nb_[0|1][0|1][0|1]|slate|dplasma
#
# where `dlaf_nb_[0|1][0|1][0|1]` is `dlaf_nb_<hpx queue><non-blocking mechanism><mpi pool>`, for example
#       `dlaf_nb_111` : dlaf with non-blocking MPI(nb), shared priority queue(1), polling (1) and a dedicated core for an MPI pool (1).
#
# rpn: ranks per node
#
def trsm(lib, build_dir, nodes, rpn, m_sz, n_sz, mb_sz, nruns):
    _check_ranks_per_node(rpn)

    total_ranks = nodes * rpn
    cpus_per_rank = 72 // rpn
    gr, gc = _sq_factor(total_ranks)

    if lib.startswith("dlaf"):
        extra_flags = _dlaf_flags(lib)
        cmd = f"{build_dir}/miniapp/miniapp_triangular_solver --m {m_sz} --n {n_sz} --mb {mb_sz} --nb {mb_sz} --grid-rows {gr} --grid-cols {gc} --nruns {nruns} --hpx:use-process-mask {extra_flags}"
    elif lib == "slate":
        cmd = f"{build_dir}/test/slate_test trsm --dim {m_sz}x{n_sz}x0 --nb {mb_sz} --p {gr} --q {gc} --repeat {nruns} --alpha 2 --check n --ref n --type d >> trsm_{lib}.out"
    elif lib == "dplasma":
        if rpn != 1:
            ValueError("DPLASMA can only run with 1 rank per node!")
        cmd = f"{build_dir}/tests/testing_dtrsm -M {m_sz} -N {n_sz} --MB {mb_sz} --NB {mb_sz} --grid-rows {gr} --grid-cols {gc} -c 36 -v >> trsm_{lib}.out"
    else:
        raise ValueError(_err_msg(lib))

    return f"\nsrun -n {total_ranks} -c {cpus_per_rank} {cmd} >> trsm_{lib}.out"
