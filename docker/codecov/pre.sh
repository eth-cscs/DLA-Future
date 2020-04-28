#!/bin/bash

# Preprocessing: create a baseline coverage report from gcno files
# with 0 hits per line

lcov --no-external --capture --initial --base-directory /DLA-Future --directory /DLA-Future-build --output-file /shared/baseline-codecov.info &> /dev/null
