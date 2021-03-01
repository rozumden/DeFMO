#!/bin/bash

dl="${1:-./dl/nuspro}"

mkdir -p "${dl}"
# dl="$( cd "${dl}" && pwd )"
(
    cd "${dl}"
    gdrive download -r --skip 0B6eYf2Rj8c79UVVIZElldzNVS1k && \
        gdrive download -r --skip 0BwFzRq8t3gu5cEFBdF9QWWJhOGM
)
