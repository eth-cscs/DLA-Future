#!/bin/bash

# Get the full paths of the test binaries
TEST_BINARIES=`ctest -L MPI -V -N | grep -oP "(?<=\")(/root/DLA-Future-build.*?)(?=\")"`

# Copy all test commands over to a single shell script
SARUS_TEST_COMMANDS=$(ctest -L MPI -V -N | sed -n -e 's/^.*Test command: //p' | sed 's|/usr/bin/srun|srun|g')

# And replace absolute paths with basenames, assuming we move all tests to $PATH
while IFS= read -r binary; do
    short=`basename "$binary"`
    SARUS_TEST_COMMANDS=`sed "s|$binary|$short|g" <<< "$SARUS_TEST_COMMANDS"`
done <<< "$TEST_BINARIES"
