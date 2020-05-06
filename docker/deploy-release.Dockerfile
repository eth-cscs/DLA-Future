ARG BUILD_ENV

FROM $BUILD_ENV as builder

# Build DLA-Future
COPY . /DLA-Future

RUN touch /usr/bin/srun && chmod +x /usr/bin/srun

RUN source /opt/intel/compilers_and_libraries/linux/mkl/bin/mklvars.sh intel64 && \
    export MKL_LIB=$MKLROOT/lib/intel64 && \
    mkdir DLA-Future-build && cd DLA-Future-build && \
    CC=/usr/local/mpich/bin/mpicc CXX=/usr/local/mpich/bin/mpicxx cmake /DLA-Future \
      -DMKL_ROOT=/opt/intel/compilers_and_libraries/linux/mkl \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DDLAF_WITH_CUDA=OFF \
      -DDLAF_WITH_MKL=ON \
      -DDLAF_WITH_TEST=ON \
      -DDLAF_BUILD_MINIAPPS=ON \
      -DDLAF_MPI_PRESET=slurm \
      -DDLAF_TEST_RUNALL_WITH_MPIEXEC=ON \
      -DMPIEXEC_NUMCORES=36 \
      -DMPIEXEC_EXECUTABLE=srun \
      -DMPIEXEC_PREFLAGS="--jobid=\$JOBID;sarus;run;--mpi;\$IMAGE" \
      -DCMAKE_INSTALL_PREFIX=/usr && \
      make -j$(nproc) && \
      source /DLA-Future/ci/sarusify.sh && \
      echo "$SARUS_TEST_COMMANDS" > /root/run.sh && \
      # Bundle into self-contained folder
      # We have to copy a couple of MKL libs that are dlopen'ed by hand
      /root/libtree/libtree \
        --chrpath \
        -d /root/DLA-Future.bundle \
        $TEST_EXECUTABLES \
        ${MKL_LIB}/libmkl_avx.so \
        ${MKL_LIB}/libmkl_avx2.so \
        ${MKL_LIB}/libmkl_core.so \
        ${MKL_LIB}/libmkl_def.so \
        ${MKL_LIB}/libmkl_intel_thread.so \
        ${MKL_LIB}/libmkl_mc.so \
        ${MKL_LIB}/libmkl_mc3.so \
        ${MKL_LIB}/libmkl_sequential.so \
        ${MKL_LIB}/libmkl_tbb_thread.so \
        ${MKL_LIB}/libmkl_vml_avx.so \
        ${MKL_LIB}/libmkl_vml_avx2.so \
        ${MKL_LIB}/libmkl_vml_cmpt.so \
        ${MKL_LIB}/libmkl_vml_def.so \
        ${MKL_LIB}/libmkl_vml_mc.so \
        ${MKL_LIB}/libmkl_vml_mc3.so && \
      rm -rf /DLA-Future

FROM ubuntu:18.04

COPY --from=builder /root/DLA-Future.bundle /root/DLA-Future.bundle
COPY --from=builder /root/run.sh /root/run.sh

# Make it easy to call our binaries.
ENV PATH="/root/DLA-Future.bundle/usr/bin:$PATH"

# Automatically print stacktraces on segfault
ENV LD_PRELOAD=/lib/x86_64-linux-gnu/libSegFault.so

RUN echo "/root/DLA-Future.bundle/usr/lib/" > /etc/ld.so.conf.d/dlaf.conf && ldconfig

WORKDIR /root/DLA-Future.bundle/usr/bin
