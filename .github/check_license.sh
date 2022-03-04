#!/usr/bin/env bash

usage() {
  echo "$0 usage: [-r n] license file"
  echo ""
  echo "where:"
  echo "-r n  max line number (1-based) where the first line of license header should start."
  echo "      First line of the header must fall within range [1,n] (default: 1)"
  echo ""
  echo "SPECIAL TAGS: they can be used inside files to control behavior of this tool"
  echo ""
  echo "dlaf-no-license-check       skip any check and return 0"
  echo "dlaf-multi-license-check    check license in the entire file (-r option is ignored)"

  exit 0
}

MAX_LINE=1
MULTILICENSE=false

while getopts "hr:" arg; do
  case $arg in
    r)
      MAX_LINE=$OPTARG
      [ $MAX_LINE -lt 1 ] && echo "line number must be a value >= 1" && exit 1
      ;;
    h | *)
      usage
      ;;
  esac
done

# note: check positional arguments
[ $((OPTIND+1)) -gt $# ] && usage

LICENSE_FILE=${@:$OPTIND:1}
FILE_TO_CHECK=${@:$OPTIND+1:1}

[ ! -f $LICENSE_FILE ]  && echo "$LICENSE_FILE not found."   && exit 1
[ ! -f $FILE_TO_CHECK ] && echo "$FILE_TO_CHECK not found."  && exit 1


# SPECIAL TAGs

# note: skip the check
grep -sq "dlaf-no-license-check" $FILE_TO_CHECK && exit 0

# note: multi-license check
if grep -sq "dlaf-multi-license-check" $FILE_TO_CHECK; then
  MAX_LINE=`cat $FILE_TO_CHECK | wc -l | tr -d ' '`
fi

# Given a "hook" to locate the license header, it is used to:
#       1. determine the offset of the hook in the license file
HOOK_STR="Copyright .* ETH Zurich"
HOOK_LINENUM=`awk -vHOOK="$HOOK_STR" '$0 ~ HOOK { print NR }' $LICENSE_FILE`
HOOK_LICENSE=`awk -vHOOK="$HOOK_STR" '$0 ~ HOOK { print $0 }' $LICENSE_FILE`
LICENSE_SIZE=$((`cat $LICENSE_FILE | wc -l` - 1))
#       2.  locate the hook inside the file to check and determine the (potential) position
#           where the license header starts in it
LICENSE_POS=$((`grep -n "$HOOK_LICENSE" $FILE_TO_CHECK | cut -d':' -f1` - HOOK_LINENUM + 1))

# License extraction
# error: if no hook is found OR it is to close to the beginning of the file
[ $LICENSE_POS -lt 1 ] && echo "LICENSE NOT FOUND" && exit 1

# note: extract potential license header lines from the file to check, and
#       compare them with the actual license text
sed -n "${LICENSE_POS},+${LICENSE_SIZE}p" $FILE_TO_CHECK | diff - $LICENSE_FILE > /dev/null
LICENSE_FOUND=$?

# Check result
[ $LICENSE_FOUND -ne 0 ]        && echo "LICENSE NOT FOUND"                       && exit 1
[ $LICENSE_POS -gt $MAX_LINE ]  && echo "LICENSE NOT IN THE RANGE [1, $MAX_LINE]" && exit 1
echo "LICENSE OK" && exit 0
