#!/usr/bin/env bash

set -euo pipefail

date=$(date -Iseconds)

benchmark_name="${1}"
shift

echo ""
echo "##########"
echo "Running benchmarks with name \"${benchmark_name}\""

# configurations=(mc gpu)
configurations=(gpu)
# configurations=(mc)

for configuration in "${configurations[@]}"; do
    echo ""
    echo "##########"
    echo "Handling configuration \"${configuration}\""

    if [[ "${configuration}" == "mc" ]]; then
        spec="dla-future%gcc@9.3.0 arch=cray-cnl7-haswell build_type=Release ~cuda ^intel-mkl"
        cmake_flags="-DCMAKE_BUILD_TYPE=Release -DDLAF_WITH_CUDA=OFF -DDLAF_WITH_MKL=ON -DDLAF_BUILD_MINIAPPS=ON"
    elif [[ "${configuration}" == "gpu" ]]; then
        spec="dla-future%gcc@9.3.0 arch=cray-cnl7-haswell build_type=Release +cuda ^intel-mkl"
        cmake_flags="-DCMAKE_BUILD_TYPE=Release -DDLAF_WITH_CUDA=ON -DDLAF_WITH_MKL=ON -DDLAF_BUILD_MINIAPPS=ON"
    else
        echo "configuration should be \"mc\" or \"gpu\" (\"${configuration}\" given)"
    fi

    paths_array=()
    for commitish in "$@"; do
        commit=$(git rev-parse ${commitish})

        # create worktree
        worktree_path=$SCRATCH/dlaf-comparisons/${commit}
        [[ ! -d "${worktree_path}" ]] && git worktree add "${worktree_path}" "${commit}"

        # create spack repository
        repo_path="${worktree_path}/spack-${commit}"
        [[ ! -L "${repo_path}" ]] && ln -s "${worktree_path}/spack" "${repo_path}"

        spack_cmd="spack -c repos:[\"${repo_path}\"]"

        spec_hash=$(${spack_cmd} spec --json ${spec} | jq --raw-output '.spec.nodes[] | select(.name == "dla-future") | .full_hash')
        cmake_flags_hash=$(echo ${cmake_flags} | sha1sum | head -c 16)
        build_dir="${worktree_path}/build/${spec_hash}/${cmake_flags_hash}"

        echo "##########"
        echo "Handling commitish ${commitish}"
        echo "    commit: ${commit}"
        echo "    spack spec: ${spec} (full hash: ${spec_hash})"
        echo "    CMake flags: ${cmake_flags} (hash: ${cmake_flags_hash})"
        echo "    worktree path: ${worktree_path}"
        echo "    build dir: ${build_dir}"

#        # clean and make build directory
#        rm -rf "${build_dir}"
#        mkdir -p "${build_dir}"
#
        ${spack_cmd} spec --fresh -lI ${spec}
#        # install dependencies
#        ${spack_cmd} install --fresh --only dependencies ${spec}
#
#        # build dla-future
#        ${spack_cmd} build-env --fresh ${spec} -- cmake ${worktree_path} -B${build_dir} ${cmake_flags}
#        ${spack_cmd} build-env --fresh ${spec} -- cmake --build ${build_dir} --target miniapp_{cholesky,triangular_solver,gen_to_std}
#
#        paths_array+=(${build_dir})
    done

#    echo ""
#    echo "##########"
#    echo "launching jobs for the following build paths:"
#    for p in "${paths_array[@]}"; do
#        echo "    ${p}"
#    done
#
#    # launch jobs
#    ./gen_strong_comparison.py --config ${configuration} --rundir ${HOME}/dlaf-benchmarks/${date}-${benchmark_name}-${configuration} --paths "${paths_array[@]}"
#    ./gen_weak_comparison.py --config ${configuration} --rundir ${HOME}/dlaf-benchmarks/${date}-${benchmark_name}-${configuration} --paths "${paths_array[@]}"
done
