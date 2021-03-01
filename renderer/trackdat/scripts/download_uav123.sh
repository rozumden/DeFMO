#!/bin/bash

dl="${1:-./dl/uav123}"

mkdir -p "${dl}"
# dl="$( cd "${dl}" && pwd )"
(
    cd "${dl}"
    gdrive download --skip 0B6sQMCU1i4NbNGxWQzRVak5yLWs
)
