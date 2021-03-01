#!/bin/bash

# Use environment variable to set year.
# For unpacking, these are only used for the default name.
VOT_YEAR="${VOT_YEAR:-2018}"
VOT_CHALLENGE="${VOT_CHALLENGE:-main}"

if [ "${VOT_CHALLENGE}" == "main" ]; then
    default_name="vot${VOT_YEAR}"
else
    default_name="vot${VOT_YEAR}_${VOT_CHALLENGE}"
fi

dl="${1:-"./dl/${default_name}"}"
data="${2:-"./data/${default_name}"}"
scripts="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

mkdir -p "${data}"
cp "${dl}/description.json" "${data}/"|| exit 1
python "$scripts/unzip_vot.py" "$dl" "$data"
