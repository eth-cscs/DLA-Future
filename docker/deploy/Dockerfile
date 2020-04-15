ARG BUILD_ENV

FROM $BUILD_ENV as builder

ENV MKL_LIB=/opt/intel/compilers_and_libraries/linux/mkl/lib/intel64

RUN ldconfig

# Build DLA-Future
COPY . /DLA-Future

RUN touch /usr/bin/srun && chmod +x /usr/bin/srun

SHELL ["/bin/bash", "-c"]

RUN mkdir DLA-Future-build && cd DLA-Future-build && \
    CC=/usr/local/mpich/bin/mpicc CXX=/usr/local/mpich/bin/mpicxx cmake /DLA-Future \
      -DMKL_ROOT=/opt/intel/compilers_and_libraries/linux/mkl \
      -DCMAKE_BUILD_TYPE=Release \
      -DDLAF_WITH_CUDA=ON \
      -DDLAF_WITH_MKL=ON \
      -DDLAF_WITH_TEST=ON \
      -DDLAF_BUILD_MINIAPPS=OFF \
      -DDLAF_MPI_PRESET=slurm \
      -DMPIEXEC_NUMCORES=32 \
      -DMPIEXEC_EXECUTABLE=srun \
      -DMPIEXEC_PREFLAGS="--jobid=\$JOBID;unbuffer;sarus;run;-t;--mpi;\$IMAGE" \
      -DCMAKE_INSTALL_PREFIX=/usr && \
      source /DLA-Future/ci/sarusify.sh && \
      echo "$SARUS_TEST_COMMANDS" > /root/run.sh && \
      make -j$(nproc) && \
      # Let's just bundle the libs to make the docker image small
      # We have to copy a couple of MKL libs by hand
      /root/libtree/libtree --strip --chrpath \
        -d /root/DLA-Future.bundle \
        $TEST_BINARIES \
        ${MKL_LIB}/libmkl_tbb_thread.so \
        ${MKL_LIB}/libmkl_intel_thread.so \
        ${MKL_LIB}/libmkl_sequential.so \
        ${MKL_LIB}/libmkl_def.so \
        ${MKL_LIB}/libmkl_vml_def.so \
        ${MKL_LIB}/libmkl_vml_cmpt.so \
        ${MKL_LIB}/libmkl_mc.so \
        ${MKL_LIB}/libmkl_vml_mc.so \
        ${MKL_LIB}/libmkl_mc3.so \
        ${MKL_LIB}/libmkl_vml_mc3.so \
        ${MKL_LIB}/libmkl_avx.so \
        ${MKL_LIB}/libmkl_vml_avx.so \
        ${MKL_LIB}/libmkl_avx2.so \
        ${MKL_LIB}/libmkl_vml_avx2.so \
        ${MKL_LIB}/libmkl_avx512_mic.so \
        ${MKL_LIB}/libmkl_vml_avx512_mic.so \
        ${MKL_LIB}/libmkl_avx512.so \
        ${MKL_LIB}/libmkl_vml_avx512.so \
        ${MKL_LIB}/libmkl_core.so && \
      rm -rf /DLA-Future

FROM ubuntu:18.04

# This is the only thing necessary really from nvidia/cuda's ubuntu18.04 runtime image
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=10.1 brand=tesla,driver>=384,driver<385 brand=tesla,driver>=396,driver<397 brand=tesla,driver>=410,driver<411"

COPY --from=builder /root/DLA-Future.bundle /root/DLA-Future.bundle
COPY --from=builder /root/run.sh /root/run.sh

# Make it easy to call our binaries.
ENV PATH="/root/DLA-Future.bundle/usr/bin:$PATH"

RUN echo "/root/DLA-Future.bundle/usr/lib/" > /etc/ld.so.conf.d/cosma.conf && ldconfig

WORKDIR /root/DLA-Future.bundle/usr/bin
