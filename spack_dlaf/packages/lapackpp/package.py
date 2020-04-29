from spack import *

class Lapackpp(CMakePackage):
    """LAPACK++: C++ API for the Basic Linear Algebra Subroutines (University of Tennessee)"""

    homepage = "https://bitbucket.org/icl/lapackpp"
    hg       = "https://bitbucket.org/icl/lapackpp"
    maintainers = ['Sely85', 'teonnik']

    version('develop', hg=hg, revision="7ffa486")

    depends_on('blaspp')

    def cmake_args(self):
        return ['-DBUILD_LAPACKPP_TESTS=OFF']
