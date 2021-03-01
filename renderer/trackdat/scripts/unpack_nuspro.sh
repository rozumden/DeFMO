#!/bin/bash

dl="${1:-./dl/nuspro}"
data="${2:-./data/nuspro}"

mkdir -p "${data}"
dl="$( cd "${dl}" && pwd )"
# data="$( cd "${data}" && pwd )"
(
    cd "${data}"
    ls "${dl}"/data/*.zip | xargs -t -n 1 unzip -q -o
)
