FROM docker.io/rocm/dev-ubuntu-22.04:6.0.2

COPY lib64 /usr/lib
COPY packages.yaml /root/.spack/packages.yaml
COPY alps-cluster-config/site /root/site
