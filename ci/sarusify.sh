#!/bin/bash

# Copy all test commands over to a single shell script
TEST_COMMANDS=$(ctest -V -N | sed -n -e 's/^.*Test command: //p' | sed 's|/usr/bin/srun|srun|g')
SARUS_TEST_COMMANDS=""
TEST_EXECUTABLES=""

# And replace absolute paths with basenames, assuming we move all tests to $PATH
while IFS= read -r TEST_COMMAND; do
    FILTERED_COMMAND=""

    # Wrap non-MPI tests in `srun .. sarus run ...`
    if [[ $TEST_COMMAND != srun* ]]; then
        FILTERED_COMMAND='srun "-n" "1" "-c" "36" "--jobid=$JOBID" "sarus" "run" "$IMAGE" '
    fi

    for arg in $TEST_COMMAND; do
        # Remove leading and trailing quotes
        var="${arg%\"}"
        var="${var#\"}"

        # Collect the test executables and
        # replace absolute paths with basenames
        if [[ $var == /root/DLA-Future-build/* ]]; then
            TEST_EXECUTABLES="$var"$'\n'"$TEST_EXECUTABLES"
            FILTERED_COMMAND="$FILTERED_COMMAND\"$(basename "$var")\" "
        else
            FILTERED_COMMAND="$FILTERED_COMMAND$arg "
        fi
    done

    SARUS_TEST_COMMANDS="$FILTERED_COMMAND"$'\n'"$SARUS_TEST_COMMANDS"
done <<< "$TEST_COMMANDS"
