from itertools import product
from math import sqrt
from os import makedirs, system
from os.path import expanduser, isfile
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
    if rpn <= 0:
        raise ValueError(f"Wrong value rpn = {rpn}!")
    if lib == "scalapack":
        return
    if not rpn in system["Allowed rpns"]:
        raise ValueError(f"Wrong value rpn = {rpn}!")
    if rpn != 1 and lib == "dplasma":
        raise ValueError("DPLASMA can only run with 1 rank per node!")


# return a dict with item "nodes" if rpn==None
# return a dict with items "nodes", "rpn", "total_ranks", "cores_per_rank", "threads_per_rank" otherwise.
def _computeResourcesNeeded(system, nodes, rpn):
    resources = {"nodes": nodes}
    if rpn != None:
        resources["rpn"] = rpn
        resources["total_ranks"] = nodes * rpn
        resources["cores_per_rank"] = system["Cores"] // rpn
        resources["threads_per_rank"] = system["Threads per core"] * resources["cores_per_rank"]
    return resources


# return the list containing the values of total_ranks, cores_per_rank, threads_per_rank.
def _computeResourcesNeededList(system, nodes, rpn):
    resources = _computeResourcesNeeded(system, nodes, rpn)
    return [resources["total_ranks"], resources["cores_per_rank"], resources["threads_per_rank"]]


def _err_msg(lib):
    return f"No such `lib`: {lib}! Allowed values are : `dlaf`, `slate`, `dplasma` and `scalapack` (cholesky only)."


class JobText:

    # creates the preamble of a job script and store it in self.job_text
    # The preamble is created using the template in system["Batch preamble"] where the
    # following parameters are replaced with their actual value (using python format command):
    # Always replaced: {run_name}, {time_min}, {bs_name}, {nodes}
    # Only if rpn!=None: {rpn}, {total_ranks}, {cores_per_rank}, {threads_per_rank}
    # Note: rpn!=None is required if system["Multiple rpn in same job"]==False
    def __init__(self, system, run_name, nodes, time_min, bs_name="job", rpn=None):
        self.system = system
        self.nodes = nodes
        self.bs_name = bs_name
        self.rpn = rpn

        if rpn == None and not system["Multiple rpn in same job"]:
            raise ValueError(
                'rpn has to be specified for systems with "Multiple rpn in same job": False'
            )

        subs = {
            "run_name": run_name,
            "time_min": time_min,
            "bs_name": bs_name,
            **self._computeResourcesNeeded(rpn),
        }
        if "Extra subs" in system:
            subs = system["Extra subs"](subs)

        self.job_text = system["Batch preamble"].format(**subs).strip()

    # append to the self.job_text a command of the form {env} {run_command} {command}
    # where run_command is system["Run_command"] where the following parameters
    # are replaced with the actual values (using python format command):
    # {nodes}, {rpn}, {total_ranks}, {cores_per_rank}, {threads_per_rank}
    # and env command are the strings returned by command_gen(system=self.system, nodes=self.nodes, **args)
    # Note: if rpn was None at initialization any value for rpn can be given
    #       otherwise rpn has to match with the valeu wgiven at initialization.
    def addCommand(self, command_gen, **args):
        rpn = args["rpn"]
        if self.rpn != None and self.rpn != rpn:
            raise ValueError(
                "rpn ({}) doesn't match with the value given in the initialization ({}).".format(
                    rpn, self.rpn
                )
            )

        subs = self._computeResourcesNeeded(rpn)
        if "Extra subs" in self.system:
            subs = self.system["Extra subs"](subs)

        run_cmd = self.system["Run command"].format(**subs).strip()
        [command, env] = command_gen(system=self.system, nodes=self.nodes, **args)

        self.job_text += "\n" + f"{env} {run_cmd} {command}".strip()
        if "sleep" in self.system:
            self.job_text += "\nsleep {}".format(self.system["sleep"])

    # Creates the job directory tree and the job script.
    # If debug is False it submits the job as well.
    def submitJobs(self, run_dir, debug=False):
        nodes = self.nodes
        bs_name = self.bs_name
        job_path = expanduser(f"{run_dir}/{nodes}")
        makedirs(job_path, exist_ok=True)
        job_file = f"{job_path}/{bs_name}.sh"
        with open(job_file, "w") as f:
            f.write(self.job_text + "\n")

        if debug:
            print(f"Created : {job_file}")
            return

        print(f"Submitting : {job_file}")
        system(f"sbatch --chdir={job_path} {job_file}")
        # sleep to not overload the scheduler
        sleep(1)

    # return a dict with item "nodes" if rpn==None
    # return a dict with items "nodes", "rpn", "total_ranks", "cores_per_rank", "threads_per_rank" otherwise.
    def _computeResourcesNeeded(self, rpn):
        return _computeResourcesNeeded(self.system, self.nodes, rpn)


def _checkAppExec(fname):
    if not isfile(fname):
        raise RuntimeError(f"Executable {fname} doesn't exist")


# lib: allowed libraries are dlaf|slate|dplasma|scalapack
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
    [total_ranks, cores_per_rank, threads_per_rank] = _computeResourcesNeededList(system, nodes, rpn)
    grid_cols, grid_rows = _sq_factor(total_ranks)

    if lib.startswith("dlaf"):
        env += " OMP_NUM_THREADS=1"
        app = f"{build_dir}/miniapp/miniapp_cholesky"
        opts = f"--matrix-size {m_sz} --block-size {mb_sz} --grid-rows {grid_rows} --grid-cols {grid_cols} --nruns {nruns} --pika:use-process-mask {extra_flags}"
    elif lib == "slate":
        env += f" OMP_NUM_THREADS={cores_per_rank}"
        app = f"{build_dir}/test/tester"
        if system["GPU"]:
            extra_flags += " --origin d --target d"
        opts = f"--dim {m_sz}x{m_sz}x0 --nb {mb_sz} --p {grid_rows} --q {grid_cols} --repeat {nruns} --check n --ref n --type d {extra_flags} potrf"
    elif lib == "dplasma":
        env += " OMP_NUM_THREADS=1"
        app = f"{build_dir}/tests/testing_dpotrf"
        if system["GPU"]:
            extra_flags += " -g 1"
        opts = f"-N {m_sz} --MB {mb_sz} --NB {mb_sz} --grid-rows {grid_rows} --grid-cols {grid_cols} -c {cores_per_rank} --nruns {nruns} -v {extra_flags}"
    elif lib == "scalapack":
        env += f" OMP_NUM_THREADS={cores_per_rank}"
        app = f"{build_dir}/cholesky"
        opts = f"-N {m_sz} -b {mb_sz} --p_grid={grid_rows},{grid_cols} -r {nruns} {extra_flags}"
    else:
        raise ValueError(_err_msg(lib))

    _checkAppExec(app)
    cmd = f"{app} {opts}".strip() + f" >> chol_{lib}_{suffix}.out 2>&1"
    return cmd, env.strip()


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
    [total_ranks, cores_per_rank, threads_per_rank] = _computeResourcesNeededList(system, nodes, rpn)
    gr, gc = _sq_factor(total_ranks)

    if lib.startswith("dlaf"):
        env += " OMP_NUM_THREADS=1"
        app = f"{build_dir}/miniapp/miniapp_triangular_solver"
        opts = f"--m {m_sz} --n {n_sz} --mb {mb_sz} --nb {mb_sz} --grid-rows {gr} --grid-cols {gc} --nruns {nruns} --pika:use-process-mask {extra_flags}"
    elif lib == "slate":
        env += f" OMP_NUM_THREADS={cores_per_rank}"
        app = f"{build_dir}/test/tester"
        if system["GPU"]:
            extra_flags += " --origin d --target d"
        opts = f"--dim {m_sz}x{n_sz}x0 --nb {mb_sz} --p {gr} --q {gc} --repeat {nruns} --alpha 2 --check n --ref n --type d {extra_flags} trsm"
    elif lib == "dplasma":
        env += " OMP_NUM_THREADS=1"
        app = f"{build_dir}/tests/testing_dtrsm"
        if system["GPU"]:
            extra_flags += " -g 1"
        opts = f"-M {m_sz} -N {n_sz} --MB {mb_sz} --NB {mb_sz} --grid-rows {gr} --grid-cols {gc} -c {cores_per_rank} -v {extra_flags}"
    else:
        raise ValueError(_err_msg(lib))

    _checkAppExec(app)
    cmd = f"{app} {opts}".strip() + f" >> trsm_{lib}_{suffix}.out 2>&1"
    return cmd, env.strip()


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
    cores_per_rank = system["Cores"] // rpn
    grid_cols, grid_rows = _sq_factor(total_ranks)

    if lib.startswith("dlaf"):
        env += " OMP_NUM_THREADS=1"
        app = f"{build_dir}/miniapp/miniapp_gen_to_std"
        opts = f"--matrix-size {m_sz} --block-size {mb_sz} --grid-rows {grid_rows} --grid-cols {grid_cols} --nruns {nruns} --pika:use-process-mask {extra_flags}"
    elif lib == "slate":
        env += f" OMP_NUM_THREADS={cores_per_rank}"
        app = f"{build_dir}/test/tester"
        if system["GPU"]:
            extra_flags += " --origin d --target d"
        opts = f"--dim {m_sz}x{m_sz}x0 --nb {mb_sz} --p {grid_rows} --q {grid_cols} --repeat {nruns} --check n --ref n --type d {extra_flags} hegst"
    else:
        raise ValueError(_err_msg(lib))

    _checkAppExec(app)
    cmd = f"{app} {opts}".strip() + f" >> hegst_{lib}_{suffix}.out 2>&1"
    return cmd, env.strip()


def _dictProduct(d):
    p = [dict(zip(d.keys(), items)) for items in product(*d.values())]
    return p


class StrongScaling:
    # setup a strong scaling test
    # time has to be given in minutes.
    def __init__(self, system, run_name, batch_script_filename, nodes_arr, time):
        self.job = {
            "system": system,
            "run_name": run_name,
            "bs_name": batch_script_filename,
            "nodes_arr": nodes_arr,
            "time": time,
        }
        self.runs = []
        self.rpn_preamble = None

    # add one/multiple runs
    def add(self, miniapp, lib, build_dir, params, nruns, suffix="", extra_flags="", env=""):
        if "rpn" not in params:
            raise KeyError("params dictionary should contain the key 'rpn'")

        # convert single params in a list with a single item
        for i in params:
            if not isinstance(params[i], list):
                params[i] = [params[i]]

        if self.job["system"]["Multiple rpn in same job"] == False:
            if len(params["rpn"]) != 1:
                raise ValueError(
                    'Only a single rpn is allowed when system["Multiple rpn in same job"] is False. (Got {})'.format(
                        len(params["rpn"])
                    )
                )
            if self.rpn_preamble == None:
                self.rpn_preamble = params["rpn"][0]
            if self.rpn_preamble != params["rpn"][0]:
                raise ValueError(
                    'Cannot mix different rpn in the same script if system["Multiple rpn in same job"] is False. (Old value: {}, new value {})'.format(
                        self.run_preamble, params["rpn"][0]
                    )
                )

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
        # If no runs has been added return an empty string.
        if len(self.runs) == 0:
            return ""

        job = self.job
        job_text = JobText(
            job["system"], job["run_name"], nodes, job["time"], job["bs_name"], self.rpn_preamble
        )
        for run in self.runs:
            product_params = _dictProduct(run["params"])

            for param in product_params:
                rpn = param["rpn"]
                suffix = "rpn={}".format(rpn)
                if run["suffix"] != "":
                    suffix = "{}_{}".format(run["suffix"], suffix)
                job_text.addCommand(
                    run["miniapp"],
                    lib=run["lib"],
                    build_dir=run["build_dir"],
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
            print(self.jobText(nodes).job_text)
            print()

    # Create dir structure and batch scripts and (if !debug) submit
    # Post: The object is cleared and is in the state as after construction.
    def submit(self, run_dir, debug):
        for nodes in self.job["nodes_arr"]:
            job_text = self.jobText(nodes)
            job_text.submitJobs(run_dir, debug=debug)
        self.runs = []
        self.rpn_preamble = None


class WeakScaling:
    # setup a strong scaling test
    # time_0 and time has to be given in minutes.
    # job time is then computed as time_0 + time * sqrt(nodes)
    #     (This time derivation assumes a N**3 complexity, where N = N1 * sqrt(nodes), and a perfect parallel efficiency)
    def __init__(self, system, run_name, batch_script_filename, nodes_arr, time_0, time):
        self.job = {
            "system": system,
            "run_name": run_name,
            "bs_name": batch_script_filename,
            "nodes_arr": nodes_arr,
            "time_0": time_0,
            "time": time,
        }
        self.runs = []
        self.rpn_preamble = None

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

        if self.job["system"]["Multiple rpn in same job"] == False:
            if len(params["rpn"]) != 1:
                raise ValueError(
                    'Only a single rpn is allowed when system["Multiple rpn in same job"] is False. (Got {})'.format(
                        len(params["rpn"])
                    )
                )
            if self.rpn_preamble == None:
                self.rpn_preamble = params["rpn"][0]
            if self.rpn_preamble != params["rpn"][0]:
                raise ValueError(
                    'Cannot mix different rpn in the same script if system["Multiple rpn in same job"] is False. (Old value: {}, new value {})'.format(
                        self.run_preamble, params["rpn"][0]
                    )
                )

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
        # If no runs has been added return an empty string.
        if len(self.runs) == 0:
            return ""

        job = self.job
        job_text = JobText(
            job["system"], job["run_name"], nodes, self.getTime(nodes), job["bs_name"], self.rpn_preamble
        )

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
                    job_text.addCommand(
                        run["miniapp"],
                        lib=run["lib"],
                        build_dir=run["build_dir"],
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
            print(self.jobText(nodes).job_text)
            print()

    # Create dir structure and batch scripts and (if !debug) submit
    # Post: The object is cleared and is in the state as after construction.
    def submit(self, run_dir, debug):
        for nodes in self.job["nodes_arr"]:
            job_text = self.jobText(nodes)
            job_text.submitJobs(run_dir, debug=debug)
        self.runs = []
        self.rpn_preamble = None
