# Copyright 2013-2022 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# dlaf-no-license-check

from spack import *


class DlaFuture(CMakePackage, CudaPackage):
    """DLA-Future library: Distributed Linear Algebra with Future"""

    homepage = "https://github.com/eth-cscs/DLA-Future/wiki"
    git = "https://github.com/eth-cscs/DLA-Future"

    maintainers = ["teonnik", "albestro", "Sely85"]

    version("develop", branch="master")

    variant("doc", default=False, description="Build documentation.")

    variant("miniapps", default=False, description="Build miniapps.")

    variant("ci-test", default=False, description="Build for CI (Advanced usage).")
    conflicts('~miniapps', when='+ci-test')

    depends_on("cmake@3.16:", type="build")
    depends_on("doxygen", type="build", when="+doc")
    depends_on("mpi")
    depends_on("blaspp")
    depends_on("lapackpp")

    depends_on("umpire~examples")
    depends_on("umpire+cuda~shared", when="+cuda")

    # https://github.com/eth-cscs/DLA-Future/issues/420
    conflicts("umpire@6:")

    depends_on("pika cxxstd=17 +mpi")
    depends_on("pika@main")
    depends_on("pika +cuda", when="+cuda")

    depends_on("pika build_type=Debug", when="build_type=Debug")
    depends_on("pika build_type=Release", when="build_type=Release")
    depends_on("pika build_type=RelWithDebInfo", when="build_type=RelWithDebInfo")

    def cmake_args(self):
        spec = self.spec

        # BLAS/LAPACK
        if "^mkl" in spec:
            args = [self.define("DLAF_WITH_MKL", True)]
        else:
            args = [
                self.define("DLAF_WITH_MKL", False),
                self.define("LAPACK_TYPE", "Custom"),
                self.define(
                    "LAPACK_LIBRARY",
                    " ".join([spec[dep].libs.ld_flags for dep in ["blas", "lapack"]]),
                ),
            ]

        # CUDA
        args.append(self.define_from_variant("DLAF_WITH_CUDA", "cuda"))

        # DOC
        args.append(self.define_from_variant("DLAF_BUILD_DOC", "doc"))

        if '+ci-test' in self.spec:
            # Enable TESTS and setup CI specific parameters
            args.append(self.define("CMAKE_CXX_FLAGS", "-Werror"))
            args.append(self.define("BUILD_TESTING", True))
            args.append(self.define("DLAF_BUILD_TESTING", True))
            args.append(self.define("DLAF_CI_RUNNER_USES_MPIRUN", True))
            args.append(self.define("MPIEXEC_EXECUTABLE", "srun"))
        else:
            # TEST
            args.append(self.define("DLAF_BUILD_TESTING", self.run_tests))

        # MINIAPPS
        args.append(self.define_from_variant("DLAF_BUILD_MINIAPPS", "miniapps"))

        return args
