#!/bin/bash

# Use environment variable to set subsample rate.
TRACKINGNET_RATE="${TRACKINGNET_RATE:-1}"

if [ "${TRACKINGNET_RATE}" -gt 1 ]; then
    default_name="trackingnet_${TRACKINGNET_RATE}"
else
    default_name=trackingnet
fi

dl="${1:-./dl/trackingnet}"
data="${2:-"./data/${default_name}"}"

mkdir -p "${data}"
dl="$( cd "${dl}" && pwd )"
data="$( cd "${data}" && pwd )"

# Copy anno and zips from dl/data/ to data/.
# (This is inefficient, could use symlinks instead.)
rsync -av "${dl}/data/" "${data}/" || exit 1

(
    cd "${dl}/TrackingNet-devkit" || exit 1
    python extract_frame.py --trackingnet_dir "${data}" || exit 1
) || exit 1
# Then remove zips.
rm -r "${data}"/*/zips || exit 1

# Take subset of frames to reduce size.
if [ "${TRACKINGNET_RATE}" -gt 1 ]; then
    chunks="TEST $( seq 0 11 | xargs -I{} echo TRAIN_{} )"
    for chunk in $chunks; do
        (
            cd "$data/$chunk" && \
            find frames -type f -name '*.jpg' \
                | sed -e 's/\.jpg$//' \
                | awk -F/ '($3 % '"${TRACKINGNET_RATE}"' == 0) {print $2 "/" $3 ".jpg"}' \
                >subset.txt && \
            rm -rf frames_tmp && \
            mkdir frames_tmp && \
            rsync --files-from=subset.txt frames/ frames_tmp/ && \
            rm subset.txt && \
            rm -r frames && \
            mv frames_tmp frames
        ) || exit 1
    done
fi
