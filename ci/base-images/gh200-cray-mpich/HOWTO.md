# Modified base image to allow building cray-mpich

## Preparation steps

```
mkdir lib64
cp -a /usr/lib64/libcuda.* lib64/
cp -a /usr/lib64/libxpmem.* lib64/

git clone https://github.com/eth-cscs/alps-cluster-config.git

cp alps-cluster-config/daint/packages.yaml packages.yaml
```

## Edit cluster config files

Modify `packages.yaml`:
```
    xpmem:
      buildable: false
      externals:
      - spec: xpmem@2.9.6
        prefix: /usr
    libfabric:
-     buildable: false
-     externals:
-     - spec: libfabric@1.15.2.0
-       prefix: /opt/cray/libfabric/1.15.2.0/
+     require: "@1.15.2.0"
    slurm:
      buildable: false
      externals:
      - spec: slurm@23-11-7
        prefix: /usr
```
Note: The container engine (CE) will replace libfabric with the system one when running the container.
Make sure to use the same version.


Modify `alps-cluster-config/site/repo/packages/cray-gtl/package.py`
```
                 patchelf("--force-rpath", "--set-rpath", rpath, f, fail_on_error=False)
                 # The C compiler wrapper can fail because libmpi_gtl_cuda refers to the symbol
                 # __gxx_personality_v0 but wasn't linked against libstdc++.
-                if "libmpi_gtl_cuda.so" in str(f):
-                    patchelf("--add-needed", "libstdc++.so", f, fail_on_error=False)
                 if "@8.1.27+cuda" in self.spec:
                     patchelf("--add-needed", "libcudart.so", f, fail_on_error=False)
                     patchelf("--add-needed", "libcuda.so", f, fail_on_error=False)
```
Note: the library links `libstdc++.so` from version 8.1.23. All the available aarch64 libraries already link with it,
therefore we can safely remove it for gh200.

## Build and push

```
export TAG="v1.3"
CSCS_REGISTRY="jfrog.svc.cscs.ch/docker-ci-ext/4700071344751697"
podman login jfrog.svc.cscs.ch

podman build -f build.Dockerfile -t $CSCS_REGISTRY/base-images/cuda_12.6.1-devel-ubuntu24.04:$TAG

podman push $CSCS_REGISTRY/base-images/cuda_12.6.1-devel-ubuntu24.04:$TAG
```
