#!/bin/bash

dl="${1:-./dl/ilsvrc}"

mkdir -p "${dl}"
dl="$( cd "${dl}" && pwd )"

(
    cd "${dl}"
    wget -c "http://bvisionweb1.cs.unc.edu/ilsvrc2015/ILSVRC2015_VID.tar.gz"
)
