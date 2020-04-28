#!/bin/bash

# Post processing: combine all *.info reports into a single one
# and prune code cov from external or generated files

TRACE_FILES_ARGS=`find /shared -type f -iname '*.info' -exec sh -c "echo --add-tracefile {}" \;`

lcov ${TRACE_FILES_ARGS} --output-file /shared/combined.info

# Only keep our own source
lcov --extract /shared/combined.info "/DLA-Future/*" --output-file /shared/combined.info

# Upload to codecov.io
pushd /DLA-Future
bash <(curl -s https://codecov.io/bash) -f /shared/combined.info
popd
