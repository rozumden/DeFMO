#!/bin/bash

dl="${1:-./dl/alov}"

mkdir -p "${dl}"
# dl="$( cd "${dl}" && pwd )"
(
    cd "${dl}"
    wget -c "http://isis-data.science.uva.nl/alov/alov300++GT_txtFiles.zip" && \
        wget -c "http://isis-data.science.uva.nl/alov/alov300++_frames.zip"
)
