ARG BASE_IMAGE=ubuntu:24.04

FROM $BASE_IMAGE

# set jfrog autoclean policy
LABEL com.jfrog.artifactory.retention.maxDays="21"

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="$PATH:/opt/spack/bin" \
    SPACK_COLOR=always

# Overwrite entrypoint as NVIDIA images set a script that clog the output.
ENTRYPOINT []
CMD [ "/bin/bash" ]
SHELL ["/bin/bash", "-c"]

ARG EXTRA_APTGET
# python is needed for spack and fastcov
# codecov upload needs curl + ca-certificates
# glibc-tools is needed for libSegFault on ubuntu > 22.04
# jq, strace are needed for check-threads
# tzdata is needed to print correct time
RUN apt-get -yqq update && \
    apt-get -yqq install --no-install-recommends \
    software-properties-common \
    build-essential gfortran \
    autoconf automake libssl-dev ninja-build pkg-config \
    gawk git tar \
    wget curl ca-certificates gpg-agent tzdata \
    python3 python3-setuptools \
    glibc-tools jq strace \
    patchelf unzip file gnupg2 libncurses-dev \
    ${EXTRA_APTGET} && \
    rm -rf /var/lib/apt/lists/*

# These are the versions of spack and spack-packages we want to have
ARG SPACK_SHA
ENV SPACK_SHA=$SPACK_SHA
ARG SPACK_PACKAGES_SHA
ENV SPACK_PACKAGES_SHA=$SPACK_PACKAGES_SHA

# Install the specific ref of Spack provided by the user and find compilers
RUN mkdir -p /opt/spack && \
    curl -Ls "https://api.github.com/repos/spack/spack/tarball/$SPACK_SHA" | tar --strip-components=1 -xz -C /opt/spack && \
    mkdir -p /opt/spack-packages && \
    curl -Ls "https://api.github.com/repos/spack/spack-packages/tarball/$SPACK_PACKAGES_SHA" | tar --strip-components=1 -xz -C /opt/spack-packages
RUN spack repo add --scope site /opt/spack-packages/repos/spack_repo/builtin

# Find compilers + Define which compiler we want to use
ARG COMPILER
RUN spack external find gcc llvm && \
    spack config add "packages:cxx:require:'${COMPILER}'" && \
    spack config add "packages:c:require:'${COMPILER}'" && \
    spack config add "packages:fortran:require:gcc"

RUN spack external find \
    autoconf \
    automake \
    bzip2 \
    cuda \
    diffutils \
    findutils \
    git \
    ninja \
    m4 \
    ncurses \
    openssl \
    perl \
    pkg-config \
    python \
    xz

# Add our custom spack repo from here
ARG SPACK_DLAF_REPO
COPY $SPACK_DLAF_REPO /user_repo

RUN spack repo add --scope site /user_repo

### Workaround until CE provides full MPI substitution.
ARG ALPS_CLUSTER_CONFIG_SHA
ENV ALPS_CLUSTER_CONFIG_SHA=$ALPS_CLUSTER_CONFIG_SHA
RUN mkdir -p /opt/alps-cluster-config && \
    curl -Ls "https://api.github.com/repos/eth-cscs/alps-cluster-config/tarball/$ALPS_CLUSTER_CONFIG_SHA" | \
    tar --strip-components=1 -xz -C /opt/alps-cluster-config && \
    spack repo add --scope site /opt/alps-cluster-config/site/spack_repo/alps

# Set this to a spack.yaml file which contains a spec
# e.g. --build-arg SPACK_ENVIRONMENT=ci/spack/my-env.yaml
ARG SPACK_ENVIRONMENT
ARG COMMON_SPACK_ENVIRONMENT
ARG ENV_VIEW=/view

# Build dependencies
# 1. Create a spack environment named `ci` from the input spack.yaml file
COPY $SPACK_ENVIRONMENT /spack_environment/spack.yaml
COPY $COMMON_SPACK_ENVIRONMENT /spack_environment/
RUN spack env create --with-view ${ENV_VIEW} ci /spack_environment/spack.yaml
# 2. Set the C++ standard
ARG CXXSTD=20
RUN spack -e ci config add "packages:dla-future:variants:cxxstd=${CXXSTD}"
# 3. Concretize environment
RUN spack -e ci concretize
# 4. Install only the dependencies of this (top level is our package)
ARG NUM_PROCS
RUN spack -e ci install --jobs ${NUM_PROCS} --fail-fast --only=dependencies

# make ctest executable available.
RUN ln -s `spack -e ci location -i cmake`/bin/ctest /usr/bin/ctest

RUN echo ${ENV_VIEW}/lib > /etc/ld.so.conf.d/dlaf.conf && ldconfig
