'''

Expects directory structure:
    {video}/{frame:05d}.jpg
    {video}/groundtruth.txt
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from . import dataset
from . import util


def load_nuspro(dir):
    return dataset.load_csv_dataset_simple(
        dir, _discover_tracks, _annot_file, _image_file,
        fieldnames=['xmin', 'ymin', 'xmax', 'ymax'],
        init_time=1, delim=' ')


def _discover_tracks(dir):
    subdirs = util.list_subdirs(dir)
    return [v for v in subdirs if os.path.isfile(os.path.join(dir, _annot_file(v)))]


def _annot_file(video_id):
    return os.path.join(video_id, 'groundtruth.txt')


def _image_file(video_id):
    return os.path.join(video_id, '{:05d}.jpg')
