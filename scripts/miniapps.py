from os import system, makedirs
from os.path import expanduser
from re import sub
from time import sleep

# Finds two factors of `n` that are as close to each other as possible.
#
# Note: the second factor is larger or equal to the first factor
def _sq_factor(n):
    for i in range(1, int(n ** 0.5) + 1):
        if n % i == 0:
            f = (i, n // i)
    return f


def _check_ranks_per_node(system, lib, rpn):
    if not rpn in system["Allowed rpns"]:
        raise ValueError(f"Wrong value rpn = {rpn}!")
    if rpn != 1 and lib == "dplasma":
        raise ValueError("DPLASMA can only run with 1 rank per node!")


def _err_msg(lib):
    return f"No such `lib`: {lib}! Allowed values are : `dlaf`, `slate` and `dplasma`."


# Job preamble
#
def init_job_text(system, run_name, nodes, time_min):
    return system["Batch preamble"].format(run_name=run_name, nodes=nodes, time_min=time_min).strip()


# Run command with options
#
def run_command(system, total_ranks, cpus_per_rank):
    threads_per_rank = system["Threads per core"] * cpus_per_rank
    return system["Run command"].format(total_ranks=total_ranks, cpus_per_rank=cpus_per_rank, threads_per_rank=threads_per_rank)


# Create the job directory tree and submit jobs.
#
def submit_jobs(run_dir, nodes, job_text, suffix="na"):
    job_path = expanduser(f"{run_dir}/{nodes}")
    makedirs(job_path, exist_ok=True)
    job_file = f"{job_path}/job_{suffix}.sh"
    with open(job_file, "w") as f:
        f.write(job_text)

    print(f"Submitting : {job_file}")
    system(f"sbatch --chdir={job_path} {job_file}")
    # sleep to not overload the scheduler
    sleep(1)


# lib: allowed libraries are dlaf|slate|dplasma
# rpn: ranks per node
#
def chol(system, lib, build_dir, nodes, rpn, m_sz, mb_sz, nruns, suffix="na", extra_flags="", env=""):
    _check_ranks_per_node(system, lib, rpn)

    total_ranks = nodes * rpn
    cpus_per_rank = system["Cores"] // rpn
    grid_cols, grid_rows = _sq_factor(total_ranks)

    if lib.startswith("dlaf"):
        env += " OMP_NUM_THREADS=1"
        cmd = f"{build_dir}/miniapp/miniapp_cholesky --matrix-size {m_sz} --block-size {mb_sz} --grid-rows {grid_rows} --grid-cols {grid_cols} --nruns {nruns} --hpx:use-process-mask {extra_flags}"
    elif lib == "slate":
        env += f" OMP_NUM_THREADS={cpus_per_rank}"
        cmd = f"{build_dir}/test/tester --dim {m_sz}x{m_sz}x0 --nb {mb_sz} --p {grid_rows} --q {grid_cols} --repeat {nruns} --check n --ref n --type d {extra_flags} potrf"
    elif lib == "dplasma":
        env += " OMP_NUM_THREADS=1"
        cmd = f"{build_dir}/tests/testing_dpotrf -N {m_sz} --MB {mb_sz} --NB {mb_sz} --grid-rows {grid_rows} --grid-cols {grid_cols} -c {cpus_per_rank} --nruns {nruns} -v {extra_flags}"
    else:
        raise ValueError(_err_msg(lib))

    run_cmd = run_command(system, total_ranks, cpus_per_rank)
    return (
        "\n" + f"{env} {run_cmd} {cmd} >> chol_{lib}_{suffix}.out 2>&1".strip()
    )


# lib: allowed libraries are dlaf|slate|dplasma
# rpn: ranks per node
#
def trsm(
    system, lib, build_dir, nodes, rpn, m_sz, n_sz, mb_sz, nruns, suffix="na", extra_flags="", env=""
):
    _check_ranks_per_node(system, lib, rpn)

    total_ranks = nodes * rpn
    cpus_per_rank = system["Cores"] // rpn
    gr, gc = _sq_factor(total_ranks)

    if lib.startswith("dlaf"):
        env += " OMP_NUM_THREADS=1"
        cmd = f"{build_dir}/miniapp/miniapp_triangular_solver --m {m_sz} --n {n_sz} --mb {mb_sz} --nb {mb_sz} --grid-rows {gr} --grid-cols {gc} --nruns {nruns} --hpx:use-process-mask {extra_flags}"
    elif lib == "slate":
        env += f" OMP_NUM_THREADS={cpus_per_rank}"
        cmd = f"{build_dir}/test/tester --dim {m_sz}x{n_sz}x0 --nb {mb_sz} --p {gr} --q {gc} --repeat {nruns} --alpha 2 --check n --ref n --type d {extra_flags} trsm"
    elif lib == "dplasma":
        env += " OMP_NUM_THREADS=1"
        cmd = f"{build_dir}/tests/testing_dtrsm -M {m_sz} -N {n_sz} --MB {mb_sz} --NB {mb_sz} --grid-rows {gr} --grid-cols {gc} -c {cpus_per_rank} -v {extra_flags}"
    else:
        raise ValueError(_err_msg(lib))

    run_cmd = run_command(system, total_ranks, cpus_per_rank)
    return (
        "\n" + f"{env} {run_cmd} {cmd} >> trsm_{lib}_{suffix}.out 2>&1".strip()
    )
