ARG BUILD_ENV

FROM $BUILD_ENV as builder

RUN ldconfig

# Build DLA-Future
COPY . /DLA-Future

RUN touch /usr/bin/srun && chmod +x /usr/bin/srun

SHELL ["/bin/bash", "-c"]

RUN mkdir DLA-Future-build && cd DLA-Future-build && \
    CC=/usr/local/mpich/bin/mpicc CXX=/usr/local/mpich/bin/mpicxx cmake /DLA-Future \
      -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CXX_FLAGS_DEBUG="-g -Og -fno-omit-frame-pointer" \
      -DLAPACK_CUSTOM_TYPE=Custom \
      -DLAPACK_CUSTOM_INCLUDE_DIR=/usr/local/include \
      -DLAPACK_CUSTOM_LIBRARY=openblas \
      -DDLAF_WITH_CUDA=OFF \
      -DDLAF_WITH_MKL=OFF \
      -DDLAF_WITH_TEST=ON \
      -DDLAF_BUILD_MINIAPPS=OFF \
      -DDLAF_MPI_PRESET=slurm \
      -DMPIEXEC_NUMCORES=32 \
      -DMPIEXEC_EXECUTABLE=srun \
      -DMPIEXEC_PREFLAGS="--jobid=\$JOBID;sarus;run;--mpi;\$IMAGE" \
      -DCMAKE_INSTALL_PREFIX=/usr && \
      source /DLA-Future/ci/sarusify.sh && \
      echo "$SARUS_TEST_COMMANDS" > /root/run.sh && \
      make -j$(nproc) && \
      # Let's just bundle the libs to make the docker image small
      # We have to copy a couple of MKL libs by hand
      /root/libtree/libtree \
        --chrpath \
        -d /root/DLA-Future.bundle \
        $TEST_BINARIES && \
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
