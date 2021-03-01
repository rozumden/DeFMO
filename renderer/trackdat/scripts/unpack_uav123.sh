#!/bin/bash

dl="${1:-./dl/uav123}"
data="${2:-./data/uav123}"

mkdir -p "${data}"
dl="$( cd "${dl}" && pwd )"
# data="$( cd "${data}" && pwd )"
(
    cd "${data}"
    unzip -o "${dl}/Dataset_UAV123.zip"
)
