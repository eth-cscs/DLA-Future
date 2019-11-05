# Copyright 2013-2019 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# ----------------------------------------------------------------------------
# If you submit this package back to Spack as a pull request,
# please first remove this boilerplate and all FIXME comments.
#
# This is a template package file for Spack.  We've put "FIXME"
# next to all the things you'll want to change. Once you've handled
# them, you can save this file and test your package like this:
#
#     spack install lapackpp
#
# You can edit this file again by typing:
#
#     spack edit lapackpp
#
# See the Spack documentation for more information on packaging.
# ----------------------------------------------------------------------------

from spack import *

class Lapackpp(CMakePackage):
    """LAPACK++: C++ API for the Basic Linear Algebra Subroutines (University of Tennessee)"""

    # FIXME: Add a proper url for your package's homepage here.
    homepage = "https://bitbucket.org/icl/lapackpp"
    hg       = "https://bitbucket.org/icl/lapackpp"

    maintainers = ['Sely85']

    version('develop')

    # FIXME: Add proper versions and checksums here.
    # version('1.2.3', '0123456789abcdef0123456789abcdef')

    # FIXME: Add dependencies if required.
    depends_on('intel-mkl')
    depends_on('libtest')
    depends_on('blaspp')

    #def cmake_args(self):
        # FIXME: Add arguments other than
        # FIXME: CMAKE_INSTALL_PREFIX and CMAKE_BUILD_TYPE
        # FIXME: If not needed delete this function
    #    args = [']
    #    return args

