#!/bin/bash

dl="${1:-./dl/otb}"
data="${2:-./data/otb}"

mkdir -p "${data}"
dl="$( cd "${dl}" && pwd )"
# data="$( cd "${data}" && pwd )"
(
    cd "${data}"
    ls "${dl}"/videos/*.zip | xargs -t -n 1 unzip -q -o
)
