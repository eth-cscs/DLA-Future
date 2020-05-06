#!/bin/bash

# Create a random name for the coverage report to avoid clashes in mpi
name=`cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 32 | head -n 1`

# Execute the test
$@

# Create the report
lcov --no-external --capture --base-directory /DLA-Future --directory /DLA-Future-build --output-file /shared/${name}.info &> /dev/null
