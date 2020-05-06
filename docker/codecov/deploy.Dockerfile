# Build environment image
ARG BUILD_ENV

# This is the folder where the project is built
ARG BUILD=/DLA-Future-build

# This is where we copy the sources to
ARG SOURCE=/DLA-Future

# Where "make install" should go / we copy the bare minimum
# of binaries to here
ARG DEPLOY=/root/DLA-Future.bundle

FROM $BUILD_ENV as builder

ARG BUILD
ARG SOURCE
ARG DEPLOY

SHELL ["/bin/bash", "-c"]

# Build DLA-Future
COPY . $SOURCE

# ctest wants to be able to locate the test binaries when
# you run `ctest -N`, so fake an srun command here.
RUN touch /usr/bin/srun && chmod +x /usr/bin/srun

# Build the project with coverage symbols
RUN mkdir ${BUILD} && cd ${BUILD} && \
    CC=/usr/local/mpich/bin/mpicc CXX=/usr/local/mpich/bin/mpicxx cmake ${SOURCE} \
      -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CXX_FLAGS_DEBUG="-g -O0 -fprofile-arcs -ftest-coverage" \
      -DCMAKE_EXE_LINKER_FLAGS="-fprofile-arcs -ftest-coverage" \
      -DLAPACK_CUSTOM_TYPE=Custom \
      -DLAPACK_CUSTOM_INCLUDE_DIR=/usr/local/include \
      -DLAPACK_CUSTOM_LIBRARY=openblas \
      -DDLAF_WITH_CUDA=OFF \
      -DDLAF_WITH_MKL=OFF \
      -DDLAF_WITH_TEST=ON \
      -DDLAF_BUILD_MINIAPPS=ON \
      -DDLAF_MPI_PRESET=slurm \
      -DDLAF_TEST_RUNALL_WITH_MPIEXEC=ON \
      -DMPIEXEC_NUMCORES=36 \
      -DMPIEXEC_EXECUTABLE=srun \
      # wrap all commands in sarus and run.sh
      -DMPIEXEC_PREFLAGS="--jobid=\$JOBID;sarus;run;--mpi;--mount=type=bind,source=\$PWD,destination=/shared;\$IMAGE;bash;-xe;${SOURCE}/docker/codecov/run.sh" \
      -DCMAKE_INSTALL_PREFIX=/usr && \
      make -j$(nproc) && \
      source ${SOURCE}/ci/sarusify.sh && \
      echo "$SARUS_TEST_COMMANDS" > /root/run.sh && \
      /root/libtree/libtree \
        --chrpath \
        -d ${DEPLOY} \
        $(which gcov) \
        $(which addr2line) \
        $TEST_EXECUTABLES && \
      # Copy lcov over (it's perl scripts, so cannot use libtree)
      cp -L $(which lcov) $(which geninfo) ${DEPLOY}/usr/bin && \
      # Remove everything except for gcno coverage files
      mv ${BUILD} ${BUILD}-tmp && \
      mkdir ${BUILD} && \
      cd ${BUILD}-tmp && \
      find -iname "*.gcno" -exec cp --parent \{\} ${BUILD} \; && \
      rm -rf ${BUILD}-tmp

# Multistage build, this is the final small image
FROM ubuntu:18.04

ARG BUILD
ARG SOURCE
ARG DEPLOY

# Install perl to make lcov happy
# codecov upload needs curl + ca-certificates
# TODO: remove git after https://github.com/codecov/codecov-bash/pull/291
#       or https://github.com/codecov/codecov-bash/pull/265 is merged
RUN apt-get update && \
    apt-get install --no-install-recommends -qq \
      perl \
      curl \
      ca-certificates \
      git && \
    rm -rf /var/lib/apt/lists/*

# Copy the executables and the codecov gcno files
COPY --from=builder ${BUILD} ${BUILD}
COPY --from=builder ${DEPLOY} ${DEPLOY}

# Copy the source files into the image as well.
# This is necessary for code coverage of MPI tests: gcov has to have write temporary
# data into the source folder. In distributed applications we can therefore not mount
# the git repo folder at runtime in the container, because it is shared and would
# cause race conditions in gcov. When PR #291 or #265 (see above) is merged
# we can at the very least remove all remnants of git, in particular the `.git` folder...
COPY --from=builder ${SOURCE} ${SOURCE}

# Finally copy the srun commands
COPY --from=builder /root/run.sh /root/run.sh

# Make it easy to call our binaries.
ENV PATH="${DEPLOY}/usr/bin:$PATH"

# Automatically print stacktraces on segfault
ENV LD_PRELOAD=/lib/x86_64-linux-gnu/libSegFault.so

RUN echo "${DEPLOY}/usr/lib/" > /etc/ld.so.conf.d/dlaf.conf && ldconfig

WORKDIR ${DEPLOY}/usr/bin
