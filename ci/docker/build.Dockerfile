ARG BASE_IMAGE=ubuntu:20.04

FROM $BASE_IMAGE

# set jfrog autoclean policy
LABEL com.jfrog.artifactory.retention.maxDays="21"

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="$PATH:/opt/spack/bin:/opt/libtree" \
    SPACK_COLOR=always
SHELL ["/bin/bash", "-c"]

ARG EXTRA_APTGET
RUN apt-get -yqq update && \
    apt-get -yqq install --no-install-recommends \
    software-properties-common \
    build-essential gfortran \
    autoconf automake \
    ${EXTRA_APTGET} \
    gawk \
    python3 python3-distutils \
    git tar wget curl ca-certificates gpg-agent jq tzdata \
    patchelf unzip file gnupg2 && \
    rm -rf /var/lib/apt/lists/*

# Install cmake
RUN wget -qO- "https://cmake.org/files/v3.22/cmake-3.22.1-linux-x86_64.tar.gz" | tar --strip-components=1 -xz -C /usr/local

# Install libtree for packaging
RUN mkdir -p /opt/libtree && \
    curl -Lfso /opt/libtree/libtree https://github.com/haampie/libtree/releases/download/v2.0.0/libtree_x86_64 && \
    chmod +x /opt/libtree/libtree

# Install MKL and remove static libs (to keep image smaller)
ARG USE_MKL=ON
ARG MKL_VERSION=2020.4-912
ARG MKL_SPEC=2020.4.304
RUN if [ "$USE_MKL" = "ON" ]; then \
      wget -qO - https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB 2>/dev/null | apt-key add - && \
      apt-add-repository 'deb https://apt.repos.intel.com/mkl all main' && \
      apt-get install -y -qq --no-install-recommends intel-mkl-64bit-${MKL_VERSION} && \
      rm -rf /var/lib/apt/lists/* && \
      find "/opt/intel/" -name "*.a" -delete && \
      echo -e "/opt/intel/lib/intel64\n/opt/intel/compilers_and_libraries/linux/mkl/lib/intel64" >> /etc/ld.so.conf.d/intel.conf && \
      ldconfig ; \
    fi

# This is the spack version we want to have
ARG SPACK_SHA
ENV SPACK_SHA=$SPACK_SHA

# Install the specific ref of Spack provided by the user and find compilers
RUN mkdir -p /opt/spack && \
    curl -Ls "https://api.github.com/repos/spack/spack/tarball/$SPACK_SHA" | tar --strip-components=1 -xz -C /opt/spack

# Find compilers + Add gfortran to clang specs + Define which compiler we want to use
ARG COMPILER
RUN spack compiler find && \
    gawk -i inplace '$0 ~ "compiler:" {flag=0} $0 ~ "spec:.*clang" {flag=1} flag == 1 && $1 ~ "^f[c7]" && $2 ~ "null" {gsub("null","/usr/bin/gfortran",$0)} {print $0}' /root/.spack/linux/compilers.yaml && \
    spack config add "packages:all:compiler:[${COMPILER}]"

RUN spack external find \
    autoconf \
    automake \
    bzip2 \
    cmake \
    cuda \
    diffutils \
    findutils \
    git \
    m4 \
    openssl \
    perl \
    python \
    xz && \
    if [ "$USE_MKL" = "ON" ]; then \
      echo -e "  intel-mkl:\n    externals:\n    - spec: \"intel-mkl@$MKL_SPEC\"\n      prefix: /opt/intel\n    buildable: False" >> ~/.spack/packages.yaml ; \
    fi

# Set up the binary cache and trust the public part of our signing key
COPY ./ci/docker/spack.cloud_key.asc ./spack.cloud_key.asc
RUN spack mirror add --scope site cscs https://spack.cloud && \
    spack gpg trust ./spack.cloud_key.asc

# Add our custom spack repo from here
ARG SPACK_DLAF_REPO
COPY $SPACK_DLAF_REPO /user_repo

RUN spack repo add --scope site /user_repo

# Set this to a spack.yaml file which contains a spec
# e.g. --build-arg SPACK_ENVIRONMENT=ci/spack/my-env.yaml
ARG SPACK_ENVIRONMENT
# Build dependencies
# 1. Create a spack environment named `ci` from the input spack.yaml file
# 2. Install only the dependencies of this (top level is our package)
COPY $SPACK_ENVIRONMENT /spack_environment/spack.yaml
RUN spack env create --without-view ci /spack_environment/spack.yaml
RUN spack -e ci install --fail-fast --only=dependencies
