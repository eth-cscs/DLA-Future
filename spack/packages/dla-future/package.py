# Copyright 2013-2023 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# dlaf-no-license-check

from spack.package import *


class DlaFuture(CMakePackage, CudaPackage, ROCmPackage):
    """DLA-Future library: Distributed Linear Algebra with Future"""

    homepage = "https://github.com/eth-cscs/DLA-Future"
    url = "https://github.com/eth-cscs/DLA-Future/archive/v0.0.0.tar.gz"
    git = "https://github.com/eth-cscs/DLA-Future.git"
    maintainers = ["rasolca", "albestro", "msimberg", "aurianer"]

    version("0.1.0", sha256="f7ffcde22edabb3dc24a624e2888f98829ee526da384cd752b2b271c731ca9b1")
    version("master", branch="master")

    variant("shared", default=True, description="Build shared libraries.")

    variant("hdf5", default=False, description="HDF5 support for dealing with matrices on disk.")

    variant("doc", default=False, description="Build documentation.")

    variant("miniapps", default=False, description="Build miniapps.")

    variant("scalapack", default=False, description="Build C API compatible with ScaLAPACK")

    depends_on("cmake@3.22:", type="build")
    depends_on("doxygen", type="build", when="+doc")
    depends_on("mpi")
    depends_on("blaspp@2022.05.00:")
    depends_on("lapackpp@2022.05.00:")
    depends_on("scalapack", when="+scalapack")

    depends_on("umpire~examples")
    depends_on("umpire~cuda", when="~cuda")
    depends_on("umpire~rocm", when="~rocm")
    depends_on("umpire+cuda~shared", when="+cuda")
    depends_on("umpire+rocm~shared", when="+rocm")
    depends_on("umpire@4.1.0:")

    depends_on("pika@0.16:")
    depends_on("pika-algorithms@0.1:")
    depends_on("pika +mpi")
    depends_on("pika +cuda", when="+cuda")
    depends_on("pika +rocm", when="+rocm")

    conflicts("^pika cxxstd=20", when="+cuda")

    depends_on("whip +cuda", when="+cuda")
    depends_on("whip +rocm", when="+rocm")

    depends_on("rocblas", when="+rocm")
    depends_on("rocprim", when="+rocm")
    depends_on("rocsolver", when="+rocm")
    depends_on("rocthrust", when="+rocm")

    depends_on("hdf5 +cxx+mpi+threadsafe+shared", when="+hdf5")

    conflicts("+cuda", when="+rocm")

    with when("+rocm"):
        for val in ROCmPackage.amdgpu_targets:
            depends_on("pika amdgpu_target={0}".format(val), when="amdgpu_target={0}".format(val))
            depends_on(
                "rocsolver amdgpu_target={0}".format(val), when="amdgpu_target={0}".format(val)
            )
            depends_on(
                "rocblas amdgpu_target={0}".format(val), when="amdgpu_target={0}".format(val)
            )
            depends_on(
                "rocprim amdgpu_target={0}".format(val), when="amdgpu_target={0}".format(val)
            )
            depends_on(
                "rocthrust amdgpu_target={0}".format(val), when="amdgpu_target={0}".format(val)
            )
            depends_on("whip amdgpu_target={0}".format(val), when="amdgpu_target={0}".format(val))
            depends_on(
                "umpire amdgpu_target={0}".format(val), when="amdgpu_target={0}".format(val)
            )

    with when("+cuda"):
        for val in CudaPackage.cuda_arch_values:
            depends_on("pika cuda_arch={0}".format(val), when="cuda_arch={0}".format(val))
            depends_on("umpire cuda_arch={0}".format(val), when="cuda_arch={0}".format(val))

    ### Variants available only in the DLAF repo spack package
    cxxstds = ("17", "20")
    variant(
        "cxxstd",
        default="17",
        values=cxxstds,
        description="Use the specified C++ standard when building",
    )
    conflicts("cxxstd=20", when="+cuda")

    for cxxstd in cxxstds:
        depends_on("pika cxxstd={0}".format(cxxstd), when="cxxstd={0}".format(cxxstd))
        depends_on("pika-algorithms cxxstd={0}".format(cxxstd), when="cxxstd={0}".format(cxxstd))

    variant("ci-test", default=False, description="Build for CI (Advanced usage).")
    conflicts("~miniapps", when="+ci-test")

    variant(
        "ci-check-threads",
        default=False,
        description="Check number of spawned threads in CI (Advanced usage).",
    )
    ###

    def cmake_args(self):
        spec = self.spec
        args = []

        args.append(self.define_from_variant("BUILD_SHARED_LIBS", "shared"))

        # BLAS/LAPACK
        if "^mkl" in spec:
            vmap = {
                "none": "seq",
                "openmp": "omp",
                "tbb": "tbb",
            }  # Map MKL variants to LAPACK target name
            mkl_threads = vmap[spec["intel-mkl"].variants["threads"].value]
            # TODO: Generalise for intel-oneapi-mkl
            args += [
                self.define("DLAF_WITH_MKL", True),
                self.define("MKL_LAPACK_TARGET", f"mkl::mkl_intel_32bit_{mkl_threads}_dyn"),
            ]
            if "+scalapack" in spec:
                if (
                    "^mpich" in spec
                    or "^cray-mpich" in spec
                    or "^intel-mpi" in spec
                    or "^mvapich" in spec
                    or "^mvapich2" in spec
                ):
                    mkl_mpi = "mpich"
                elif "^openmpi" in spec:
                    mkl_mpi = "ompi"
                args.append(
                    self.define(
                        "MKL_SCALAPACK_TARGET",
                        f"mkl::scalapack_{mkl_mpi}_intel_32bit_{mkl_threads}_dyn",
                    )
                )
        else:
            args.append(self.define("DLAF_WITH_MKL", False))
            args.append(
                self.define(
                    "LAPACK_LIBRARY",
                    " ".join([spec[dep].libs.ld_flags for dep in ["blas", "lapack"]]),
                )
            )
            if "+scalapack" in spec:
                args.append(self.define("SCALAPACK_LIBRARY", spec["scalapack"].libs.ld_flags))

        if "+scalapack" in spec:
            args.append(self.define_from_variant("DLAF_WITH_SCALAPACK", "scalapack"))

        # CUDA/HIP
        args.append(self.define_from_variant("DLAF_WITH_CUDA", "cuda"))
        args.append(self.define_from_variant("DLAF_WITH_HIP", "rocm"))
        if "+rocm" in spec:
            archs = self.spec.variants["amdgpu_target"].value
            if "none" not in archs:
                arch_str = ";".join(archs)
                args.append(self.define("CMAKE_HIP_ARCHITECTURES", arch_str))
        if "+cuda" in spec:
            archs = self.spec.variants["cuda_arch"].value
            if "none" not in archs:
                arch_str = ";".join(archs)
                args.append(self.define("CMAKE_CUDA_ARCHITECTURES", arch_str))

        # HDF5 support
        args.append(self.define_from_variant("DLAF_WITH_HDF5", "hdf5"))

        # DOC
        args.append(self.define_from_variant("DLAF_BUILD_DOC", "doc"))

        ### For the spack repo only the else branch should remain.
        if "+ci-test" in self.spec:
            # Enable TESTS and setup CI specific parameters
            args.append(self.define("CMAKE_CXX_FLAGS", "-Werror"))
            if "+cuda" in self.spec:
                args.append(self.define("CMAKE_CUDA_FLAGS", "-Werror=all-warnings"))
            if "+rocm" in self.spec:
                args.append(self.define("CMAKE_HIP_FLAGS", "-Werror"))
            args.append(self.define("BUILD_TESTING", True))
            args.append(self.define("DLAF_BUILD_TESTING", True))
            args.append(self.define("DLAF_BUILD_TESTING_HEADER", True))
            args.append(self.define("DLAF_CI_RUNNER_USES_MPIRUN", True))
        else:
            # TEST
            args.append(self.define("DLAF_BUILD_TESTING", self.run_tests))

        ### Variants available only in the DLAF repo spack package
        if "+ci-check-threads" in self.spec:
            args.append(self.define("DLAF_TEST_PREFLAGS", "check-threads"))
        ###

        # MINIAPPS
        args.append(self.define_from_variant("DLAF_BUILD_MINIAPPS", "miniapps"))

        return args
