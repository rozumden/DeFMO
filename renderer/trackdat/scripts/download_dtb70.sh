#!/bin/bash

dl="${1:-./dl/dtb70}"

mkdir -p "${dl}"
# dl="$( cd "${dl}" && pwd )"
(
    cd "${dl}"
    wget -c "https://www.dropbox.com/s/s1fj99s2six4lrs/DTB70.tar.gz?dl=1" -O DTB70.tar.gz
)
