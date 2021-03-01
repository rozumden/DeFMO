#!/bin/bash

# Invokes download_TrackingNet.py

dl="${1:-./dl/trackingnet}"

mkdir -p "${dl}"
dl="$( cd "${dl}" && pwd )"

# Download to dl/data/
mkdir -p "${dl}/data/"

(
    cd "${dl}"
    # Clone devkit.
    rm -rf TrackingNet-devkit || exit 1
    git clone https://github.com/SilvioGiancola/TrackingNet-devkit.git || exit 1
    # The script download_TrackingNet.py must be invoked from its directory.
    # It expects to find csv_link/*
    cd TrackingNet-devkit
    python download_TrackingNet.py --trackingnet_dir "${dl}/data/" || exit 1
) || exit 1
