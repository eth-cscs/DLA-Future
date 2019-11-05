from spack import *
import os

class Blaspp(CMakePackage):
    """BLAS++: C++ API for the Basic Linear Algebra Subroutines (University of Texas)."""

    # FIXME: Add a proper url for your package's homepage here.
    homepage = "https://bitbucket.org/icl/blaspp"

    maintainers = ['Sely85']

    version('develop')

    # FIXME: Add proper versions and checksums here.
    # version('1.2.3', '0123456789abcdef0123456789abcdef')

    # FIXME: Add dependencies if required.
    depends_on('intel-mkl')

