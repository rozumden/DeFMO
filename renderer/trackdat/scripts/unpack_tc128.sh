#!/bin/bash

dl="${1:-./dl/tc128}"
data="${2:-./data/tc128}"

mkdir -p "${data}"
dl="$( cd "${dl}" && pwd )"
# data="$( cd "${data}" && pwd )"
(
    cd "${data}"
    unzip -o "${dl}/Temple-color-128.zip"
)
