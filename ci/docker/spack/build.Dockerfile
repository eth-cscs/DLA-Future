ARG BASE_IMAGE=ubuntu:20.04

FROM $BASE_IMAGE

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="$PATH:/opt/spack/bin:/opt/libtree" \
    SPACK_COLOR=always
SHELL ["/bin/bash", "-c"]

RUN apt-get -yqq update && \
    apt-get -yqq install --no-install-recommends \
    software-properties-common \
    build-essential \
    autoconf automake \
    clang \
    python3 python3.8-distutils \
    git tar wget curl ca-certificates gpg-agent jq tzdata \
    patchelf unzip file gnupg2 && \
    rm -rf /var/lib/apt/lists/*

# Install cmake
RUN wget -qO- "https://cmake.org/files/v3.22/cmake-3.22.1-linux-x86_64.tar.gz" | tar --strip-components=1 -xz -C /usr/local

# Install libtree for packaging
RUN mkdir -p /opt/libtree && \
    curl -Lfso /opt/libtree/libtree https://github.com/haampie/libtree/releases/download/v2.0.0/libtree_x86_64 && \
    chmod +x /opt/libtree/libtree

# This is the spack version we want to have
ARG SPACK_SHA
ENV SPACK_SHA=$SPACK_SHA

# Install the specific ref of Spack provided by the user and find compilers
RUN mkdir -p /opt/spack && \
    curl -Ls "https://api.github.com/repos/spack/spack/tarball/$SPACK_SHA" | tar --strip-components=1 -xz -C /opt/spack

# Define which compiler we want to use
ARG COMPILER
RUN spack compiler find && spack config add "packages:all:compiler:[${COMPILER}]"

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
    xz

# Set up the binary cache and trust the public part of our signing key
COPY ./ci/docker/spack/public_key.asc ./public_key.asc
RUN spack mirror add --scope site cscs https://spack.cloud && \
    spack gpg trust ./public_key.asc

# Add our custom spack repo from here
COPY ./spack /user_repo

RUN spack repo add --scope site /user_repo

# Set this to a spack.yaml file which contains a spec
# e.g. --build-arg SPACK_ENVIRONMENT=ci/spack/my-env.yaml
ARG SPACK_ENVIRONMENT
# Build dependencies
# 1. Create a spack environment named `ci` from the input spack.yaml file
# 2. Install only the dependencies of this (top level is our package)
COPY $SPACK_ENVIRONMENT /spack_environment/spack.yaml
RUN spack env create --without-view ci /spack_environment/spack.yaml
RUN spack -e ci install --fail-fast --only=dependencies --require-full-hash-match
