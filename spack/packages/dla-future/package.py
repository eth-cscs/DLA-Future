# Copyright 2013-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack import *


class DlaFuture(CMakePackage):
    """DLA-Future library: Distributed Linear Algebra with Future"""

    homepage = "https://github.com/eth-cscs/DLA-Future/wiki"
    git      = "https://github.com/eth-cscs/DLA-Future"

    maintainers = ['teonnik', 'albestro', 'Sely85']

    version('develop', branch='master')

    variant('cuda', default=False,
            description='Use the GPU/cuBLAS back end.')
    variant('doc', default=False,
            description='Build documentation.')

    depends_on('cmake@3.14:', type='build')
    depends_on('doxygen', type='build', when='+doc')

    depends_on('mpi')
    depends_on('blaspp')
    depends_on('lapackpp')
    depends_on('hpx@1.4.0:1.4.1 cxxstd=14 networking=none')
    depends_on('cuda', when='+cuda')

    def cmake_args(self):
        spec = self.spec

        # BLAS/LAPACK
        if '^mkl' in spec:
            args = [ self.define('DLAF_WITH_MKL', True) ]
        else:
            args = [
                    self.define('DLAF_WITH_MKL', False),
                    self.define('LAPACK_TYPE', 'Custom'),
                    self.define('LAPACK_LIBRARY',
                        ' '.join([spec[dep].libs.ld_flags for dep in ['blas', 'lapack']]))
                   ]

        # CUDA
        args.append(self.define_from_variant('DLAF_WITH_CUDA', 'cuda'))

        # DOC
        args.append(self.define_from_variant('BUILD_DOC', 'doc'))

        # TESTs
        args.append(self.define('DLAF_WITH_TEST', self.run_tests))

        return args
