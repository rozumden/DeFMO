'''

Expects directory structure:
    {video}/img/{frame:04d}.jpg
    {video}/{video}_gt.txt
    {video}/{video}_frames.txt
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

from . import dataset
from . import util


def load_tc128(dir, keep_prev=False):
    # Cannot use load_csv_dataset_simple because
    # init frame is specified in different file.

    video_ids = _discover_tracks(dir, keep_prev=keep_prev)
    if len(video_ids) == 0:
        raise RuntimeError('no tracks found')

    labels_pix = {}
    for video_id in video_ids:
        # Necessary to be case-insensitive.
        frames_file = util.imatch_basename(os.path.join(dir, _frames_file(video_id)))
        with open(frames_file, 'r') as f:
            init_time, _ = _read_frame_range(f)
        gt_file = os.path.join(dir, _annot_file(video_id))
        with open(gt_file, 'r') as f:
            labels_pix[video_id] = dataset.load_rects_csv(
                f, fieldnames=['xmin', 'ymin', 'width', 'height'],
                init_time=init_time, delim=',')

    labels, aspects = dataset.convert_relative(dir, video_ids, labels_pix, _image_file)
    return dataset.Dataset(
        track_ids=video_ids, labels=labels,
        image_files=util.func_dict(video_ids, _image_file),
        aspects=aspects)


def _discover_tracks(dir, keep_prev=False):
    subdirs = util.list_subdirs(os.path.join(dir, 'Temple-color-128'))
    # Check if annotation file exists.
    video_ids = [subdir for subdir in subdirs
                 if os.path.isfile(os.path.join(dir, _annot_file(subdir)))]
    if not keep_prev:
        pattern = re.compile('_ce')
        video_ids = list(filter(pattern.search, video_ids))
    return video_ids


def _annot_file(video_id):
    return os.path.join('Temple-color-128', video_id, '{}_gt.txt'.format(video_id))


def _frames_file(video_id):
    return os.path.join('Temple-color-128', video_id, '{}_frames.txt'.format(video_id))


def _read_frame_range(f):
    return map(int, f.read().strip().split(','))


def _image_file(video_id):
    return os.path.join('Temple-color-128', video_id, 'img', '{:04d}.jpg')
