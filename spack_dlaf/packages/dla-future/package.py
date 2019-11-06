from spack import *

class DlaFuture(CMakePackage):
    """The DLAF package provides DLA-Future library: Distributed Linear Algebra with Future"""

    homepage = "https://github.com/eth-cscs/DLA-Future.git/wiki"
    git      = "https://github.com/eth-cscs/DLA-Future.git"

    maintainers = ['Sely85']

    version('develop', branch='master')

    variant('gpu', default=False,
            description='Use the GPU/cuBLAS back end.')

    # Until mpich is default comment this out
    #depends_on('mpi@3:')
    depends_on('mpich')
    depends_on('intel-mkl')
    depends_on('blaspp')
    depends_on('lapackpp')
    depends_on('hpx cxxstd=14 networking=none')
    depends_on('cuda', when='gpu=True')

    def cmake_args(self):
       spec = self.spec
       args = ['-DDLAF_WITH_MKL=ON']

       if '+gpu' in spec:
           args.append('-DDLAF_WITH_CUDA=ON')

       return args
