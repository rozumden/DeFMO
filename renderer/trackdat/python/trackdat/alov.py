'''

Expects directory structure:
    imagedata++/{category}/{video}/{frame:08d}.jpg
    alov300++_rectangleAnnotation_full/{category}/{video}.ann
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

from . import dataset
from . import util


def load_alov(dir):
    return dataset.load_csv_dataset_simple(
        dir, _discover_tracks, _annot_file, _image_file,
        fieldnames=['time', 'x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3'],
        delim=' ')


def _discover_tracks(dir):
    images_dir = os.path.join(dir, 'imagedata++')
    categories = filter(re.compile('\d\d-[a-zA-Z]+$').match, util.list_subdirs(images_dir))
    videos = []
    for category in categories:
        videos.extend(filter(lambda subdir: subdir.startswith(category + '_video'),
                             util.list_subdirs(os.path.join(images_dir, category))))
    return videos


def _annot_file(video_id):
    category = _category_from_video_id(video_id)
    return os.path.join('alov300++_rectangleAnnotation_full', category, video_id + '.ann')


def _image_file(video_id):
    category = _category_from_video_id(video_id)
    return os.path.join('imagedata++', category, video_id, '{:08d}.jpg')


def _category_from_video_id(video_id):
    parts = video_id.split('_video')
    if len(parts) != 2:
        raise ValueError('not a valid video id: {}'.format(video_id))
    return parts[0]
