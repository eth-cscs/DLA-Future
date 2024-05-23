# Copyright 2013-2024 Lawrence Livermore National Security, LLC and other
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

    license("BSD-3-Clause")

    version("0.4.1", sha256="ba95f26475ad68da1f3a24d091dc1b925525e269e4c83c1eaf1d37d29b526666")
    version("0.4.0", sha256="34fd0da0d1a72b6981bed0bba029ba0947e0d0d99beb3e0aad0a478095c9527d")
    version("0.3.1", sha256="350a7fd216790182aa52639a3d574990a9d57843e02b92d87b854912f4812bfe")
    version("0.3.0", sha256="9887ac0b466ca03d704a8738bc89e68550ed33509578c576390e98e76b64911b")
    version("0.2.1", sha256="4c2669d58f041304bd618a9d69d9879a42e6366612c2fc932df3894d0326b7fe")
    version("0.2.0", sha256="da73cbd1b88287c86d84b1045a05406b742be924e65c52588bbff200abd81a10")
    version("0.1.0", sha256="f7ffcde22edabb3dc24a624e2888f98829ee526da384cd752b2b271c731ca9b1")
    version("master", branch="master")

    variant("shared", default=True, description="Build shared libraries.")

    variant(
        "hdf5",
        default=False,
        when="@0.2.0:",
        description="HDF5 support for dealing with matrices on disk.",
    )

    variant("doc", default=False, description="Build documentation.")

    variant("miniapps", default=False, description="Build miniapps.")

    variant(
        "scalapack",
        default=False,
        when="@0.2.0:",
        description="Build C API compatible with ScaLAPACK",
    )

    variant(
        "mpi-gpu-aware",
        default=False,
        when="@master",
        description="Use GPU-aware MPI.",
    )
    conflicts(
        "+mpi-gpu-aware",
        when="~cuda ~rocm",
        msg="GPU-aware MPI requires +cuda or +rocm",
    )

    variant(
        "mpi-gpu-force-contiguous",
        default=True,
        when="@master +mpi-gpu-aware",
        description="Force communication buffers to be contiguous before communicating.",
    )

    generator("ninja")

    depends_on("cmake@3.22:", type="build")
    depends_on("pkgconfig", type="build")
    depends_on("doxygen", type="build", when="+doc")
    depends_on("mpi")

    depends_on("blas")
    depends_on("lapack")
    depends_on("scalapack", when="+scalapack")
    depends_on("blaspp@2022.05.00:")
    depends_on("lapackpp@2022.05.00:")
    depends_on("intel-oneapi-mkl +cluster", when="^[virtuals=scalapack] intel-oneapi-mkl")

    conflicts("intel-oneapi-mkl", when="@:0.3")

    depends_on("umpire~examples")
    depends_on("umpire~cuda", when="~cuda")
    depends_on("umpire~rocm", when="~rocm")
    depends_on("umpire+cuda~shared", when="+cuda")
    depends_on("umpire+rocm~shared", when="+rocm")
    depends_on("umpire@4.1.0:")

    depends_on("pika@0.15.1:", when="@0.1")
    depends_on("pika@0.16:", when="@0.2.0")
    depends_on("pika@0.17:", when="@0.2.1")
    depends_on("pika@0.18:", when="@0.3")
    depends_on("pika@0.19.1:", when="@0.4.0:")
    depends_on("pika-algorithms@0.1:", when="@:0.2")
    depends_on("pika +mpi")
    depends_on("pika +cuda", when="+cuda")
    depends_on("pika +rocm", when="+rocm")

    conflicts("^pika cxxstd=20", when="+cuda")

    depends_on("whip +cuda", when="+cuda")
    depends_on("whip +rocm", when="+rocm")

    depends_on("rocblas", when="+rocm")
    depends_on("rocsolver", when="+rocm")

    depends_on("rocprim", when="@:0.3 +rocm")
    depends_on("rocthrust", when="@:0.3 +rocm")

    # nvcc 11.2 and older is unable to detect fmt::formatter specializations.
    # DLA-Future 0.3.1 includes a workaround to avoid including fmt in device
    # code:
    # https://github.com/pika-org/pika/issues/870
    # https://github.com/eth-cscs/DLA-Future/pull/1045
    conflicts("^fmt@10:", when="@:0.3.0 +cuda ^cuda@:11.2")

    # Compilation problem triggered by the bundled fmt in Umpire together with
    # fmt 10, which only happens with GCC 9 and nvcc 11.2 and older:
    # https://github.com/eth-cscs/DLA-Future/issues/1044
    conflicts("^fmt@10:", when="@:0.3.0 %gcc@9 +cuda ^cuda@:11.2 ^umpire@2022.10:")

    # Pedantic warnings, triggered by GCC 9 and 10, are always errors until 0.3.1:
    # https://github.com/eth-cscs/DLA-Future/pull/1043
    conflicts("%gcc@9:10", when="@:0.3.0")

    depends_on("hdf5 +cxx+mpi+threadsafe+shared", when="+hdf5")

    conflicts("+cuda", when="+rocm")

    with when("+rocm"):
        for arch in ROCmPackage.amdgpu_targets:
            depends_on(f"pika amdgpu_target={arch}", when=f"amdgpu_target={arch}")
            depends_on(f"rocsolver amdgpu_target={arch}", when=f"amdgpu_target={arch}")
            depends_on(f"rocblas amdgpu_target={arch}", when=f"amdgpu_target={arch}")
            depends_on(f"whip amdgpu_target={arch}", when=f"amdgpu_target={arch}")
            depends_on(f"umpire amdgpu_target={arch}", when=f"amdgpu_target={arch}")

    with when("@:0.3 +rocm"):
        for arch in ROCmPackage.amdgpu_targets:
            depends_on(f"rocprim amdgpu_target={arch}", when=f"amdgpu_target={arch}")
            depends_on(f"rocthrust amdgpu_target={arch}", when=f"amdgpu_target={arch}")

    with when("+cuda"):
        for arch in CudaPackage.cuda_arch_values:
            depends_on(f"pika cuda_arch={arch}", when=f"cuda_arch={arch}")
            depends_on(f"umpire cuda_arch={arch}", when=f"cuda_arch={arch}")

    patch(
        "https://github.com/eth-cscs/DLA-Future/pull/1063/commits/efc9c176a7a8c512b3f37d079dec8c25ac1b7389.patch?full_index=1",
        sha256="7f382c872d89f22da1ad499e85ffe9881cc7404c8465e42877a210a09382e2ea",
        when="@:0.3 %gcc@13:",
    )

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
        depends_on(f"pika cxxstd={cxxstd}", when=f"cxxstd={cxxstd}")
        depends_on(f"pika-algorithms cxxstd={cxxstd}", when=f"@:0.2 cxxstd={cxxstd}")

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
        if spec["lapack"].name in INTEL_MATH_LIBRARIES:
            mkl_provider = spec["lapack"].name

            vmap = {
                "intel-oneapi-mkl": {
                    "threading": {
                        "none": "sequential",
                        "openmp": "gnu_thread",
                        "tbb": "tbb_thread",
                    },
                    "mpi": {"intel-mpi": "intelmpi", "mpich": "mpich", "openmpi": "openmpi"},
                },
                "intel-mkl": {
                    "threading": {"none": "seq", "openmp": "omp", "tbb": "tbb"},
                    "mpi": {"intel-mpi": "mpich", "mpich": "mpich", "openmpi": "ompi"},
                },
            }

            if mkl_provider not in vmap.keys():
                raise RuntimeError(
                    f"dla-future does not support {mkl_provider} as lapack provider"
                )
            mkl_mapper = vmap[mkl_provider]

            mkl_threads = mkl_mapper["threading"][spec[mkl_provider].variants["threads"].value]
            if mkl_provider == "intel-oneapi-mkl":
                args += [
                    self.define("DLAF_WITH_MKL", True),
                    self.define("MKL_INTERFACE", "lp64"),
                    self.define("MKL_THREADING", mkl_threads),
                ]
            elif mkl_provider == "intel-mkl":
                args += [
                    self.define("DLAF_WITH_MKL", True)
                    if spec.version <= Version("0.3")
                    else self.define("DLAF_WITH_MKL_LEGACY", True),
                    self.define("MKL_LAPACK_TARGET", f"mkl::mkl_intel_32bit_{mkl_threads}_dyn"),
                ]

            if "+scalapack" in spec:
                try:
                    mpi_provider = spec["mpi"].name
                    if mpi_provider in ["mpich", "cray-mpich", "mvapich", "mvapich2"]:
                        mkl_mpi = mkl_mapper["mpi"]["mpich"]
                    else:
                        mkl_mpi = mkl_mapper["mpi"][mpi_provider]
                except KeyError:
                    raise RuntimeError(
                        f"dla-future does not support {spec['mpi'].name} as mpi provider with "
                        f"the selected scalapack provider {mkl_provider}"
                    )

                if mkl_provider == "intel-oneapi-mkl":
                    args.append(self.define("MKL_MPI", mkl_mpi))
                elif mkl_provider == "intel-mkl":
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

        args.append(self.define_from_variant("DLAF_WITH_SCALAPACK", "scalapack"))

        args.append(self.define_from_variant("DLAF_WITH_MPI_GPU_AWARE", "mpi-gpu-aware"))
        args.append(self.define_from_variant("DLAF_WITH_MPI_GPU_FORCE_CONTIGUOUS", "mpi-gpu-force-contiguous"))

        # CUDA/HIP
        args.append(self.define_from_variant("DLAF_WITH_CUDA", "cuda"))
        args.append(self.define_from_variant("DLAF_WITH_HIP", "rocm"))
        if "+rocm" in spec:
            archs = spec.variants["amdgpu_target"].value
            if "none" not in archs:
                arch_str = ";".join(archs)
                args.append(self.define("CMAKE_HIP_ARCHITECTURES", arch_str))
        if "+cuda" in spec:
            archs = spec.variants["cuda_arch"].value
            if "none" not in archs:
                arch_str = ";".join(archs)
                args.append(self.define("CMAKE_CUDA_ARCHITECTURES", arch_str))

        # HDF5 support
        args.append(self.define_from_variant("DLAF_WITH_HDF5", "hdf5"))

        # DOC
        args.append(self.define_from_variant("DLAF_BUILD_DOC", "doc"))

        ### For the spack repo only the else branch should remain.
        if "+ci-test" in spec:
            # Enable TESTS and setup CI specific parameters
            args.append(self.define("CMAKE_CXX_FLAGS", "-Werror"))
            if "+cuda" in spec:
                args.append(self.define("CMAKE_CUDA_FLAGS", "-Werror=all-warnings"))
            if "+rocm" in spec:
                args.append(self.define("CMAKE_HIP_FLAGS", "-Werror"))
            args.append(self.define("BUILD_TESTING", True))
            args.append(self.define("DLAF_BUILD_TESTING", True))
            args.append(self.define("DLAF_BUILD_TESTING_HEADER", True))
            args.append(self.define("DLAF_CI_RUNNER_USES_MPIRUN", True))
        else:
            # TEST
            args.append(self.define("DLAF_BUILD_TESTING", self.run_tests))

        ### Variants available only in the DLAF repo spack package
        if "+ci-check-threads" in spec:
            args.append(self.define("DLAF_TEST_PREFLAGS", "check-threads"))
        ###

        # MINIAPPS
        args.append(self.define_from_variant("DLAF_BUILD_MINIAPPS", "miniapps"))

        return args
