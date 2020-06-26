# Copyright 2013-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack import *

class DlaFuture(CMakePackage):
    """The DLAF package provides DLA-Future library: Distributed Linear Algebra with Future"""

    homepage = "https://github.com/eth-cscs/DLA-Future.git/wiki"
    git      = "https://github.com/eth-cscs/DLA-Future.git"

    maintainers = ['teonnik', 'Sely85']

    version('develop', branch='master')

    variant('cuda', default=False,
            description='Use the GPU/cuBLAS back end.')
    variant('doc', default=False,
            description='Build documentation.')

    #depends_on('mpi@3:')
    depends_on('mpi')
    depends_on('blaspp')
    depends_on('lapackpp')
    depends_on('hpx@1.4.1: max_cpu_count=128 cxxstd=14 networking=none')
    depends_on('cuda', when='+cuda')

    def cmake_args(self):
       spec = self.spec

       if (spec.satisfies('^intel-mkl')):
           args = ['-DDLAF_WITH_MKL=ON']
       else:
           args = ['-DDLAF_WITH_MKL=OFF']
           lapack_name = spec['lapack'].libs.ld_flags.split()[1]
           blas_name = spec['blas'].libs.ld_flags.split()[1]
           lapack_libs = spec['lapack'].prefix.lib
           blas_libs = spec['blas'].prefix.lib
           args.append('-DLAPACK_TYPE=Custom')
           args.append('-DLAPACK_LIBRARY=-L{0} {1} -L{2} {3}'.format(lapack_libs, lapack_name, blas_libs, blas_name))

       if '+cuda' in spec:
           args.append('-DDLAF_WITH_CUDA=ON')

       if self.run_tests:
           args.append('-DDLAF_WITH_TEST=ON')
        else:
           args.append('-DDLAF_WITH_TEST=OFF')

       if '+doc' in spec:
           args.append('-DBUILD_DOC=on')

       return args
