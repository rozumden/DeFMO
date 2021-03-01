#!/bin/bash

dl="${1:-./dl/nfs}"
data="${2:-./data/nfs}"

mkdir -p "${data}"
dl="$( cd "${dl}" && pwd )"
# data="$( cd "${data}" && pwd )"
(
    cd "${data}"
    ls "${dl}"/*.zip | xargs -t -n 1 unzip -q -o
)
