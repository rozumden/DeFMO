from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

from PIL import Image


def infer_image_file_pattern(dir):
    elems = os.listdir(dir)
    pattern = re.compile(r'.*\.jpe?g$')
    image_files = list(filter(lambda x: pattern.match(str.lower(x)), elems))
    if not image_files:
        raise RuntimeError('no images found in directory: {}'.format(dir))
    name_ext_pairs = list(map(os.path.splitext, image_files))
    names, exts = zip(*name_ext_pairs)
    # names, exts = zip(*list(map(os.path.splitext, image_files)))
    unique_lens = set(map(len, names))
    unique_exts = set(exts)
    if len(unique_exts) > 1:
        raise RuntimeError('in dir {}: multiple extensions: {}'.format(
            dir, ', '.join(sorted(unique_exts))))
    (unique_ext,) = unique_exts
    if len(unique_lens) > 1:
        raise RuntimeError('in dir {}: multiple filename lengths: {}'.format(
            dir, ', '.join(map(str, sorted(unique_lens)))))
    (unique_len,) = unique_lens
    return ''.join(['{:0', str(unique_len), 'd}', unique_ext])
    # return ''.join(['%0', str(unique_len), 'd', unique_ext])


def imatch_basename(path):
    dir, base = os.path.split(path)
    base = imatch(dir, base)
    if base is None:
        raise RuntimeError('no case-insensitive match found: {}'.format(path))
    return os.path.join(dir, base)


def imatch(dir, base):
    # dir, base = os.path.split(path)
    matches = list(filter(lambda x: x.lower() == base.lower(), os.listdir(dir)))
    if len(matches) == 0:
        return None
    elif len(matches) == 1:
        return matches[0]
    else:
        raise RuntimeError('multiple case-insensitive matches')


def assert_file_exists(fname):
    if not os.path.isfile(fname):
        raise RuntimeError('file not found: {}'.format(fname))


def list_subdirs(dir):
    return [x for x in os.listdir(dir) if os.path.isdir(os.path.join(dir, x))]


def func_dict(keys, func):
    return {key: func(key) for key in keys}


def image_size(fname):
    # Image.open() is a lazy operation, it does not load the image.
    # https://pillow.readthedocs.io/en/5.1.x/reference/Image.html
    im = Image.open(fname)
    width, height = im.size
    return {'width': width, 'height': height}


def map_dict(f, xs):
    return dict(f(k, v) for k, v in xs.items())


def map_dict_values(f, xs):
    return {k: f(v) for k, v in xs.items()}
