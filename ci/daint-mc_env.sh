module purge
module load modules
module load craype
module load cray-mpich
module load slurm
module load xalt
module load PrgEnv-gnu
module load daint-mc
module unload cray-libsci
module load CMake
module load intel

export EASYBUILD_PREFIX=/apps/daint/SSL/rasolca/jenkins/daint-broadwell
module load EasyBuild-custom/cscs

module load BLASPP/20190829-CrayGNU-18.08
module load LAPACKPP/20190829-CrayGNU-18.08
if [ "${BUILD_TYPE,,}" = debug ]
then
  module load HPX/20190830-CrayGNU-18.08-jemalloc-debug
else
  module load HPX/20190830-CrayGNU-18.08-jemalloc
fi

export CRAYPE_LINK_TYPE=dynamic
