ARG BUILD_IMAGE
ARG DEPLOY_BASE_IMAGE

# This is the folder where the project is built
ARG BUILD=/DLA-Future-build
# This is where we copy the sources to
ARG SOURCE=/DLA-Future
# Where a bunch of shared libs live
ARG DEPLOY=/root/DLA-Future.bundle

FROM $BUILD_IMAGE as builder

ARG BUILD
ARG SOURCE
ARG DEPLOY

# Build DLA-Future
COPY . ${SOURCE}

SHELL ["/bin/bash", "-c"]

# Inject the coverage option in the spack package
RUN gawk -i inplace '$0 ~ "return args" {print "        args.append(self.define(\"DLAF_WITH_COVERAGE\", True))"} {print $0}' ${SOURCE}/spack/packages/dla-future/package.py

# Note: we force spack to build in ${BUILD} creating a link to it
RUN spack repo rm --scope site dlaf && \
    spack repo add ${SOURCE}/spack && \
    spack -e ci develop --no-clone -p ${SOURCE} dla-future@develop && \
    spack -e ci concretize -f && \
    mkdir ${BUILD} && \
    ln -s ${BUILD} `spack -e ci location -b dla-future` && \
    spack -e ci install --keep-stage

# Prune and bundle binaries
RUN mkdir ${BUILD}-tmp && cd ${BUILD} && \
    export TEST_BINARIES=`ctest --show-only=json-v1 | jq '.tests | map(.command[0]) | .[]' | tr -d \"` && \
    libtree -d ${DEPLOY} ${TEST_BINARIES} && \
    rm -rf ${DEPLOY}/usr/bin && \
    libtree -d ${DEPLOY} $(which ctest gcov addr2line) && \
    cp -L ${SOURCE}/ci/{mpi-ctest,upload_codecov} ${DEPLOY}/usr/bin && \
    echo "$TEST_BINARIES" | xargs -I{file} find -samefile {file} -exec cp --parents '{}' ${BUILD}-tmp ';' && \
    find '(' -name CTestTestfile.cmake -o -iname "*.gcno" ')' -exec cp --parents '{}' ${BUILD}-tmp ';' && \
    rm -rf ${BUILD} && \
    mv ${BUILD}-tmp ${BUILD} && \
    rm -rf ${SOURCE}/.git

# Multistage build, this is the final small image
FROM $DEPLOY_BASE_IMAGE

# set jfrog autoclean policy
LABEL com.jfrog.artifactory.retention.maxDays="7"
LABEL com.jfrog.artifactory.retention.maxCount="10"

ENV DEBIAN_FRONTEND noninteractive

ARG BUILD
ARG SOURCE
ARG DEPLOY

ARG EXTRA_APTGET_DEPLOY
# python is needed for fastcov
# pip is needed only to install fastcov (it is removed with
#     its dependencies after fastcov installation)
# codecov upload needs curl + ca-certificates
# tzdata is needed to print correct time
RUN apt-get update -qq && \
    apt-get install -qq -y --no-install-recommends \
      ${EXTRA_APTGET_DEPLOY} \
      python3 python3-pip \
      curl \
      ca-certificates \
      tzdata && \
    pip install fastcov && \
    apt-get autoremove -qq -y python3-pip && \
    apt-get clean


# Copy the executables and the codecov gcno files
COPY --from=builder ${BUILD} ${BUILD}
COPY --from=builder ${DEPLOY} ${DEPLOY}

# Copy the source files into the image as well.
# This is necessary for code coverage of MPI tests: gcov has to have write temporary
# data into the source folder. In distributed applications we can therefore not mount
# the git repo folder at runtime in the container, because it is shared and would
# cause race conditions in gcov.
COPY --from=builder ${SOURCE} ${SOURCE}

RUN cd /usr/local/bin && \
  curl -Ls https://codecov.io/bash > codecov.sh && \
  echo "f0e7a3ee76a787c37aa400cf44aee0c9b473b2fa79092edfb36d1faa853bbe23 codecov.sh" | sha256sum --check --quiet && \
  chmod +x codecov.sh

# Make it easy to call our binaries.
ENV PATH="${DEPLOY}/usr/bin:$PATH"
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=10.2"

# Used in our ctest wrapper to upload reports
ENV ENABLE_COVERAGE="YES"

# Automatically print stacktraces on segfault
ENV LD_PRELOAD=/lib/x86_64-linux-gnu/libSegFault.so

RUN echo "${DEPLOY}/usr/lib/" > /etc/ld.so.conf.d/dlaf.conf && ldconfig

WORKDIR ${BUILD}
