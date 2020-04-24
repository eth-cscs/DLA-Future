#!/bin/bash

# Copy all test commands over to a single shell script
TEST_COMMANDS=$(ctest -V -N | sed -n -e 's/^.*Test command: //p' | sed 's|/usr/bin/srun|srun|g')
SARUS_TEST_COMMANDS=""
TEST_EXECUTABLES=""

# Utilities for timers
## current time in seconds
TIMER_UTIL="alias ct=\"date +\\\"%s\\\"\""
## et(t) elapsed time since instant t (got with ct command)
TIMER_UTIL="$TIMER_UTIL"$'\n'"function et { date +%T -d \"1/1 + \$(( \`ct\` - \$1 )) sec\" }"

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
            EXECUTABLE=$(basename "$var")
            FILTERED_COMMAND="$FILTERED_COMMAND\"$EXECUTABLE\" "
        else
            FILTERED_COMMAND="$FILTERED_COMMAND$arg "
        fi
    done

    # Add check if test passed. Otherwise add it to the list of failed tests.
    FILTERED_COMMAND="$FILTERED_COMMAND"$'\n'"if [ \$? -ne 0 ]; then FAILED=\${FAILED}\"$EXECUTABLE \"; fi"

    # Add a timer for each command
    FILTERED_COMMAND="TEST_START=\$(ct)"$'\n'"$FILTERED_COMMAND"$'\n'"echo \"Elapsed: \$(et \$TEST_START)\""

    SARUS_TEST_COMMANDS="$FILTERED_COMMAND"$'\n'"$SARUS_TEST_COMMANDS"
done <<< "$TEST_COMMANDS"

# Add a timer for each command
SARUS_TEST_COMMANDS="FULL_START=\$(ct)"$'\n'"$SARUS_TEST_COMMANDS"$'\n'"echo \"Total Elapsed Time: \$(et \$TEST_START)\""

SARUS_TEST_COMMANDS="$TIMER_UTIL"'\n'"$SARUS_TEST_COMMANDS"

# Add a report with failed tests
REPORT_COMMAND="if [[ -z \$FAILED ]]
then
 echo \"Test Succeded\"
 exit 0
else
  for TEST in \$FAILED
  do
    printf \"Test %s FAILED\n\" \$TEST
  done
  exit 1
fi"

SARUS_TEST_COMMANDS="$SARUS_TEST_COMMANDS"$'\n'"$REPORT_COMMAND"
