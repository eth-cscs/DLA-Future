# Copyright 2013-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack import *


class DlaFuture(CMakePackage, CudaPackage):
    """DLA-Future library: Distributed Linear Algebra with Future"""

    homepage = "https://github.com/eth-cscs/DLA-Future/wiki"
    git = "https://github.com/eth-cscs/DLA-Future"

    maintainers = ["teonnik", "albestro", "Sely85"]

    version("develop", branch="master")

    variant("doc", default=False, description="Build documentation.")

    variant("miniapps", default=False, description="Build miniapps.")

    depends_on("cmake@3.14:", type="build")
    depends_on("doxygen", type="build", when="+doc")
    depends_on("mpi")
    depends_on("blaspp")
    depends_on("lapackpp")
    depends_on("hpx cxxstd=14 networking=none")
    depends_on("hpx@1.6.0:")
    depends_on("hpx +cuda", when="+cuda")

    depends_on("hpx build_type=Debug", when="build_type=Debug")
    depends_on("hpx build_type=Release", when="build_type=Release")
    depends_on("hpx build_type=RelWithDebInfo", when="build_type=RelWithDebInfo")

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

        # TESTs
        args.append(self.define("DLAF_BUILD_TESTING", self.run_tests))

        # MINIAPPS
        args.append(self.define_from_variant("DLAF_BUILD_MINIAPPS", "miniapps"))

        return args
