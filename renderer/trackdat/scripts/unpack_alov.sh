#!/bin/bash

dl="${1:-./dl/alov}"
data="${2:-./data/alov}"

mkdir -p "${data}"
dl="$( cd "${dl}" && pwd )"
# data="$( cd "${data}" && pwd )"
(
    cd "${data}"
    unzip -o "${dl}/alov300++GT_txtFiles.zip" && \
        unzip -o "${dl}/alov300++_frames.zip"
)
