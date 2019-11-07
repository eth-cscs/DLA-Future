from spack import *

# 1) The CMake options exposed by `blaspp` allow for a value called `auto`. The
#    value is not needed here as the choice of dependency in the spec determines
#    the appropriate flags.
# 2) BLASFinder.cmake handles most options. For `auto`, it searches all blas
#    libraries listed in `def_lib_list`.
# 3) ?? Custom blas library can be supplied via `BLAS_LIBRARIES`.
#
class Blaspp(CMakePackage):
    """BLAS++: C++ API for the Basic Linear Algebra Subroutines (University of Texas)."""

    homepage = "https://bitbucket.org/icl/blaspp"
    hg       = "https://bitbucket.org/icl/blaspp"
    maintainers = ['Sely85']

    version('develop', hg=hg)

    variant('ifort',
            default=False,
            description='Use Intel Fortran conventions. Default is GNU gfortran. (Only for Intel MKL)')

    depends_on('blas')

    def cmake_args(self):
        spec = self.spec
        args = ['-DBLASPP_BUILD_TESTS=OFF']

        # Missing:
        #
        # - acml  : BLAS_LIBRARY="AMD ACML"
        #           BLAS_LIBRARY_THREADING= threaded/sequential
        #
        # - apple : BLAS_LIBRARY="Apple Accelerate" (veclibfort ???)
        #
        if '^intel-mkl' in spec or '^intel-parallel-studio+mkl' in spec:
            args.append('-DBLAS_LIBRARY="Intel MKL"')

            # TODO: This belongs to `intel-mkl`, open a PR
            #
            if '+ifort':
                args.append('-DBLAS_LIBRARY_MKL="Intel ifort conventions"')
            else:
                args.append('-DBLAS_LIBRARY_MKL="GNU gfortran conventions"')

            if '+ilp64' in spec:
                args.append('-DBLAS_LIBRARY_INTEGER="int64_t (ILP64)"')
            else:
                args.append('-DBLAS_LIBRARY_INTEGER="int (LP64)"')

            if 'threads=openmp' in spec:
                args.append(['-DUSE_OPENMP=ON',
                             '-DBLAS_LIBRARY_THREADING="threaded"'])
            else:
                args.append('-DBLAS_LIBRARY_THREADING="sequential"')

        elif '^essl' in spec:
            args.append('-DBLAS_LIBRARY="IBM ESSL"')

            if '+ilp64' in spec:
                args.append('-DBLAS_LIBRARY_INTEGER="int64_t (ILP64)"')
            else:
                args.append('-DBLAS_LIBRARY_INTEGER="int (LP64)"')

            if 'threads=openmp' in spec:
                args.append(['-DUSE_OPENMP=ON',
                             '-DBLAS_LIBRARY_THREADING="threaded"'])
            else:
                args.append('-DBLAS_LIBRARY_THREADING="sequential"')

        elif '^openblas' in spec:
            args.append('-DBLAS_LIBRARY="OpenBLAS"')
        elif '^cray-libsci' in spec:
            args.append('-DBLAS_LIBRARY="Cray LibSci"')
        else: # e.g. netlib-lapack
            args.append('-DBLAS_LIBRARY="generic"')

        return args
