FROM docker.io/nvidia/cuda:12.6.1-devel-ubuntu24.04

COPY lib64 /usr/lib
COPY packages.yaml /root/.spack/packages.yaml
COPY alps-cluster-config/site /root/site
