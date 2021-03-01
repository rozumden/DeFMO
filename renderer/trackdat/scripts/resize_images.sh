#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 src/ dst/ size_str"
    echo
    echo "Examples for size_str:"
    echo "    '241x241!'  -- stretch to 241x241"
    echo "    '640x360>'  -- shrink to fit 640x360"
    echo "    '360x360^>' -- shrink to fill 360x360"
    exit 1
fi
src="$1"
dst="$2"
size_str="$3"
NUM_PARALLEL="${NUM_PARALLEL:-16}"

if [ ! -d "$src" ]; then
    echo "directory does not exist: $src"
    exit 1
fi

mkdir -p "$dst"

# Create directory structure.
(cd "$src" && find . -type d) | xargs -I{} mkdir -p "$dst/{}"
# Resize images.
(cd "$src" && find . -type f -iname '*.gif' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -not -name '.*') | \
    xargs -n 1 -P ${NUM_PARALLEL} -I{} convert "$src/{}" -resize "$size_str" "$dst/{}"
# Delete empty directories (except for root).
(cd "$dst" && find . -type d -mindepth 1 -empty -delete)
# Copy other files.
# (cd "$src" && find . -type f -not \( -iname '*.gif' -o -iname '*.jpg' -o -iname '*.png' -o -iname '*.jpeg' \)) | \
#     xargs -n 1 -P ${NUM_PARALLEL} -I{} cp "$src/{}" "$dst/{}"
