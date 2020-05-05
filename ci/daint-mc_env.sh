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

module load BLASPP/20190829-CrayGNU-19.10
module load LAPACKPP/20190829-CrayGNU-19.10
if [ "${BUILD_TYPE,,}" = debug ]
then
  module load HPX/1.4.0-CrayGNU-19.10-jemalloc-debug
else
  module load HPX/1.4.0-CrayGNU-19.10-jemalloc
fi

export CRAYPE_LINK_TYPE=dynamic

export CORES_PER_NODE=36
