#!/bin/bash

dl="${1:-./dl/ilsvrc}"
data="${2:-./data/ilsvrc}"

mkdir -p "${data}"

dl="$( cd "${dl}" && pwd )"
data="$( cd "${data}" && pwd )"

(
    cd "${data}"
    tar -xzf "${dl}/ILSVRC2015_VID.tar.gz"
)
