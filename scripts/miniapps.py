from itertools import product
from math import sqrt
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
    if lib == "scalapack":
        return
    if not rpn in system["Allowed rpns"]:
        raise ValueError(f"Wrong value rpn = {rpn}!")
    if rpn != 1 and lib == "dplasma":
        raise ValueError("DPLASMA can only run with 1 rank per node!")


def _err_msg(lib):
    return f"No such `lib`: {lib}! Allowed values are : `dlaf`, `slate`, `dplasma` and `scalapack` (cholesky only)."


# Job preamble
#
def init_job_text(system, run_name, nodes, time_min):
    return system["Batch preamble"].format(run_name=run_name, nodes=nodes, time_min=time_min).strip()


# Run command with options
#
def run_command(system, total_ranks, cpus_per_rank):
    threads_per_rank = system["Threads per core"] * cpus_per_rank
    return system["Run command"].format(
        total_ranks=total_ranks,
        cpus_per_rank=cpus_per_rank,
        threads_per_rank=threads_per_rank,
    )


# Create the job directory tree and submit jobs.
#
def submit_jobs(run_dir, nodes, job_text, debug=False, bs_name="job"):
    job_path = expanduser(f"{run_dir}/{nodes}")
    makedirs(job_path, exist_ok=True)
    job_file = f"{job_path}/{bs_name}.sh"
    with open(job_file, "w") as f:
        f.write(job_text + "\n")

    if debug:
        print(f"Created : {job_file}")
        return

    print(f"Submitting : {job_file}")
    system(f"sbatch --chdir={job_path} {job_file}")
    # sleep to not overload the scheduler
    sleep(1)


# lib: allowed libraries are dlaf|slate|dplasma
# rpn: ranks per node
#
def chol(
    system,
    lib,
    build_dir,
    nodes,
    rpn,
    m_sz,
    mb_sz,
    nruns,
    suffix="na",
    extra_flags="",
    env="",
):
    _check_ranks_per_node(system, lib, rpn)

    total_ranks = nodes * rpn
    cpus_per_rank = system["Cores"] // rpn
    grid_cols, grid_rows = _sq_factor(total_ranks)

    if lib.startswith("dlaf"):
        env += " OMP_NUM_THREADS=1"
        cmd = f"{build_dir}/miniapp/miniapp_cholesky --matrix-size {m_sz} --block-size {mb_sz} --grid-rows {grid_rows} --grid-cols {grid_cols} --nruns {nruns} --hpx:use-process-mask {extra_flags}"
    elif lib == "slate":
        env += f" OMP_NUM_THREADS={cpus_per_rank}"
        if system["GPU"]:
            extra_flags += " --origin d --target d"
        cmd = f"{build_dir}/test/tester --dim {m_sz}x{m_sz}x0 --nb {mb_sz} --p {grid_rows} --q {grid_cols} --repeat {nruns} --check n --ref n --type d {extra_flags} potrf"
    elif lib == "dplasma":
        env += " OMP_NUM_THREADS=1"
        if system["GPU"]:
            extra_flags += " -g 1"
        cmd = f"{build_dir}/tests/testing_dpotrf -N {m_sz} --MB {mb_sz} --NB {mb_sz} --grid-rows {grid_rows} --grid-cols {grid_cols} -c {cpus_per_rank} --nruns {nruns} -v {extra_flags}"
    elif lib == "scalapack":
        env += f" OMP_NUM_THREADS={cpus_per_rank}"
        cmd = f"{build_dir}/cholesky -N {m_sz} -b {mb_sz} --p_grid={grid_rows},{grid_cols} -r {nruns} {extra_flags}"
    else:
        raise ValueError(_err_msg(lib))

    run_cmd = run_command(system, total_ranks, cpus_per_rank)
    return "\n" + f"{env} {run_cmd} {cmd} >> chol_{lib}_{suffix}.out 2>&1".strip()


# lib: allowed libraries are dlaf|slate|dplasma
# rpn: ranks per node
# n_sz can be None in which case n_sz is set to the value of m_sz.
#
def trsm(
    system,
    lib,
    build_dir,
    nodes,
    rpn,
    m_sz,
    n_sz,
    mb_sz,
    nruns,
    suffix="na",
    extra_flags="",
    env="",
):
    if n_sz == None:
        n_sz = m_sz

    _check_ranks_per_node(system, lib, rpn)

    total_ranks = nodes * rpn
    cpus_per_rank = system["Cores"] // rpn
    gr, gc = _sq_factor(total_ranks)

    if lib.startswith("dlaf"):
        env += " OMP_NUM_THREADS=1"
        cmd = f"{build_dir}/miniapp/miniapp_triangular_solver --m {m_sz} --n {n_sz} --mb {mb_sz} --nb {mb_sz} --grid-rows {gr} --grid-cols {gc} --nruns {nruns} --hpx:use-process-mask {extra_flags}"
    elif lib == "slate":
        env += f" OMP_NUM_THREADS={cpus_per_rank}"
        if system["GPU"]:
            extra_flags += " --origin d --target d"
        cmd = f"{build_dir}/test/tester --dim {m_sz}x{n_sz}x0 --nb {mb_sz} --p {gr} --q {gc} --repeat {nruns} --alpha 2 --check n --ref n --type d {extra_flags} trsm"
    elif lib == "dplasma":
        env += " OMP_NUM_THREADS=1"
        if system["GPU"]:
            extra_flags += " -g 1"
        cmd = f"{build_dir}/tests/testing_dtrsm -M {m_sz} -N {n_sz} --MB {mb_sz} --NB {mb_sz} --grid-rows {gr} --grid-cols {gc} -c {cpus_per_rank} -v {extra_flags}"
    else:
        raise ValueError(_err_msg(lib))

    run_cmd = run_command(system, total_ranks, cpus_per_rank)
    return "\n" + f"{env} {run_cmd} {cmd} >> trsm_{lib}_{suffix}.out 2>&1".strip()


# lib: allowed libraries are dlaf|slate
# rpn: ranks per node
#
def gen2std(
    system,
    lib,
    build_dir,
    nodes,
    rpn,
    m_sz,
    mb_sz,
    nruns,
    suffix="na",
    extra_flags="",
    env="",
):
    _check_ranks_per_node(system, lib, rpn)

    total_ranks = nodes * rpn
    cpus_per_rank = system["Cores"] // rpn
    grid_cols, grid_rows = _sq_factor(total_ranks)

    if lib.startswith("dlaf"):
        env += " OMP_NUM_THREADS=1"
        cmd = f"{build_dir}/miniapp/miniapp_gen_to_std --matrix-size {m_sz} --block-size {mb_sz} --grid-rows {grid_rows} --grid-cols {grid_cols} --nruns {nruns} --hpx:use-process-mask {extra_flags}"
    elif lib == "slate":
        env += f" OMP_NUM_THREADS={cpus_per_rank}"
        if system["GPU"]:
            extra_flags += " --origin d --target d"
        cmd = f"{build_dir}/test/tester --dim {m_sz}x{m_sz}x0 --nb {mb_sz} --p {grid_rows} --q {grid_cols} --repeat {nruns} --check n --ref n --type d {extra_flags} hegst"
    else:
        raise ValueError(_err_msg(lib))

    run_cmd = run_command(system, total_ranks, cpus_per_rank)
    return "\n" + f"{env} {run_cmd} {cmd} >> hegst_{lib}_{suffix}.out 2>&1".strip()


def _dictProduct(d):
    p = [dict(zip(d.keys(), items)) for items in product(*d.values())]
    return p


class StrongScaling:
    # setup a strong scaling test
    # time has to be given in minutes.
    def __init__(self, system, run_name, nodes_arr, time):
        self.job = {
            "system": system,
            "run_name": run_name,
            "nodes_arr": nodes_arr,
            "time": time,
        }
        self.runs = []

    # add one/multiple runs
    def add(self, miniapp, lib, build_dir, params, nruns, suffix="", extra_flags="", env=""):
        if "rpn" not in params:
            raise KeyError("params dictionary should contain the key 'rpn'")

        # convert single params in a list with a single item
        for i in params:
            if not isinstance(params[i], list):
                params[i] = [params[i]]

        self.runs.append(
            {
                "miniapp": miniapp,
                "lib": lib,
                "build_dir": build_dir,
                "params": params,
                "nruns": nruns,
                "suffix": suffix,
                "extra_flags": extra_flags,
                "env": env,
            }
        )

    def jobText(self, nodes):
        job = self.job
        job_text = init_job_text(job["system"], job["run_name"], nodes, job["time"])
        for run in self.runs:
            product_params = _dictProduct(run["params"])

            for param in product_params:
                rpn = param["rpn"]
                suffix = "rpn={}".format(rpn)
                if run["suffix"] != "":
                    suffix = "{}_{}".format(run["suffix"], suffix)
                job_text += run["miniapp"](
                    system=job["system"],
                    lib=run["lib"],
                    build_dir=run["build_dir"],
                    nodes=nodes,
                    nruns=run["nruns"],
                    suffix=suffix,
                    extra_flags=run["extra_flags"],
                    env=run["env"],
                    **param,
                )
        return job_text

    # Print batch scripts
    def print(self):
        for nodes in self.job["nodes_arr"]:
            print(f"### {nodes} Nodes ###")
            print(self.jobText(nodes))
            print()

    # Create dir structure and batch scripts and (if !debug) submit
    # Post: The object is cleared and is in the state as after construction.
    def submit(self, run_dir, batch_script_filename, debug):
        for nodes in self.job["nodes_arr"]:
            job_text = self.jobText(nodes)
            submit_jobs(run_dir, nodes, job_text, debug=debug, bs_name=batch_script_filename)
        self.runs = []


class WeakScaling:
    # setup a strong scaling test
    # time_0 and time has to be given in minutes.
    # job time is then computed as time_0 + time * sqrt(nodes)
    #     (This time derivation assumes a N**3 complexity, where N = N1 * sqrt(nodes), and a perfect parallel efficiency)
    def __init__(self, system, run_name, nodes_arr, time_0, time):
        self.job = {
            "system": system,
            "run_name": run_name,
            "nodes_arr": nodes_arr,
            "time_0": time_0,
            "time": time,
        }
        self.runs = []

    def getTime(self, nodes):
        return self.job["time_0"] + round(sqrt(nodes) * self.job["time"])

    # add one/multiple runs
    # weak_params contains the parameter that will be scaled by the factor sqrt(nodes),
    #     and approximated to the near multiple of approx.
    def add(
        self,
        miniapp,
        lib,
        build_dir,
        params,
        weak_params,
        approx,
        nruns,
        suffix="",
        extra_flags="",
        env="",
    ):
        if "rpn" not in params:
            raise KeyError("params dictionary should contain the key 'rpn'")

        # convert single params in a list with a single item
        for i in params:
            if not isinstance(params[i], list):
                params[i] = [params[i]]
        for i in weak_params:
            if not isinstance(weak_params[i], list):
                weak_params[i] = [weak_params[i]]

        self.runs.append(
            {
                "miniapp": miniapp,
                "lib": lib,
                "build_dir": build_dir,
                "params": params,
                "weak_params": weak_params,
                "approx": approx,
                "nruns": nruns,
                "suffix": suffix,
                "extra_flags": extra_flags,
                "env": env,
            }
        )

    @staticmethod
    def weakScale(nodes, param, approx):
        return round(param * sqrt(nodes) / approx) * approx

    def jobText(self, nodes):
        job = self.job
        job_text = init_job_text(job["system"], job["run_name"], nodes, self.getTime(nodes))
        for run in self.runs:
            approx = run["approx"]
            product_params = _dictProduct(run["params"])
            product_weak_params = _dictProduct(run["weak_params"])

            for weak_param in product_weak_params:
                # scale weak scaling parameters
                for i in weak_param:
                    weak_param[i] = self.weakScale(nodes, weak_param[i], approx)

                for param in product_params:
                    rpn = param["rpn"]
                    suffix = "rpn={}".format(rpn)
                    if run["suffix"] != "":
                        suffix = "{}_{}".format(run["suffix"], suffix)
                    job_text += run["miniapp"](
                        system=job["system"],
                        lib=run["lib"],
                        build_dir=run["build_dir"],
                        nodes=nodes,
                        nruns=run["nruns"],
                        suffix=suffix,
                        extra_flags=run["extra_flags"],
                        env=run["env"],
                        **param,
                        **weak_param,
                    )
        return job_text

    # Print batch scripts
    def print(self):
        for nodes in self.job["nodes_arr"]:
            print(f"### {nodes} Nodes ###")
            print(self.jobText(nodes))
            print()

    # Create dir structure and batch scripts and (if !debug) submit
    # Post: The object is cleared and is in the state as after construction.
    def submit(self, run_dir, batch_script_filename, debug):
        for nodes in self.job["nodes_arr"]:
            job_text = self.jobText(nodes)
            submit_jobs(run_dir, nodes, job_text, debug=debug, bs_name=batch_script_filename)
        self.runs = []
