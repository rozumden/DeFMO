#!/bin/bash

dl="${1:-./dl/nfs}"

mkdir -p "${dl}"
# dl="$( cd "${dl}" && pwd )"
(
    cd "${dl}"
    wget -c http://ci2cv.net/nfs/Get_NFS.sh || exit 1
    bash Get_NFS.sh
)
