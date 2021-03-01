'''

Expects directory structure:
    description.json (or list.txt)
    {video}/{frame:08d}.jpg
    {video}/groundtruth.txt
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math
import os

from . import dataset
from . import util

from .vot_toolkit import vot as _vot


def load_vot(dir):
    # Cannot use load_csv_dataset_simple because
    # we use the VOT code to load the regions.
    video_ids = _load_tracks(dir)
    labels_pix = {}
    for video_id in video_ids:
        with open(os.path.join(dir, _annot_file(video_id)), 'r') as f:
            labels_pix[video_id] = _load_groundtruth(f)
    labels, aspects = dataset.convert_relative(dir, video_ids, labels_pix, _image_file)
    return dataset.Dataset(
        track_ids=video_ids, labels=labels,
        image_files=util.func_dict(video_ids, _image_file),
        aspects=aspects)


def _load_tracks(dir):
    if not (os.path.exists(os.path.join(dir, 'description.json')) or
            os.path.exists(os.path.join(dir, 'list.txt'))):
        raise RuntimeError('could not find description.json or list.txt in "{}"'.format(dir))
    try:
        with open(os.path.join(dir, 'description.json'), 'r') as f:
            description = json.load(f)
        lines = [seq['name'] for seq in description['sequences']]
    except IOError:
        with open(os.path.join(dir, 'list.txt'), 'r') as f:
            lines = f.readlines()
        # Strip whitespace and remove empty lines.
        lines = list(filter(bool, map(str.strip, lines)))
    return lines


def _annot_file(video_id):
    return os.path.join(video_id, 'groundtruth.txt')


def _image_file(video_id):
    return os.path.join(video_id, '{:08d}.jpg')


def _load_groundtruth(f, init_time=1):
    # with open(os.path.join(dir, video_id, 'groundtruth.txt'), 'r') as f:
    lines = f.readlines()
    # Strip whitespace and remove empty lines.
    lines = filter(bool, map(str.strip, lines))
    labels_pix = {}
    t = init_time
    for line in lines:
        r = _vot.convert_region(_vot.parse_region(line), 'rectangle')
        if all(math.isnan(val) for val in [r.x, r.y, r.width, r.height]):
            label = dataset.make_frame_label(absent=True)
        else:
            # TODO: Confirm that we should subtract 1 here.
            # Perhaps we should rather subtract and add 0.5 from min and max.
            rect = dataset.make_rect(xmin=r.x - 1, xmax=r.x - 1 + r.width,
                                     ymin=r.y - 1, ymax=r.y - 1 + r.height)
            label = dataset.make_frame_label(rect=rect)
        labels_pix[t] = label
        t += 1
    return labels_pix
