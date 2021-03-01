#!/bin/bash

dl="${1:-./dl/dtb70}"
data="${2:-./data/dtb70}"

mkdir -p "${data}"
dl="$( cd "${dl}" && pwd )"
# data="$( cd "${data}" && pwd )"
(
    cd "${data}"
    tar -xzf "${dl}/DTB70.tar.gz"
)
