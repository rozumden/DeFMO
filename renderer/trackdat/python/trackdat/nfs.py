'''

Expects directory structure:
    {video}/{fps}/{video}/{frame:05d}.jpg
    {video}/{fps}/{video}.txt
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

from . import util
from . import dataset


def load_nfs(dir, fps=240):
    video_ids = _discover_tracks(dir, fps)
    if len(video_ids) == 0:
        raise RuntimeError('no tracks found')

    labels_pix = {}
    for video_id in video_ids:
        annot_file = os.path.join(dir, _annot_file(video_id, fps))
        with open(annot_file, 'r') as f:
            # Ignore 'time' field because it varies between starting at 0 and 1.
            # labels_pix[video_id] = dataset.load_rects_csv(
            #     f, fieldnames=['', 'xmin', 'ymin', 'xmax', 'ymax', 'time'], delim=' ')
            labels_pix[video_id] = dataset.load_rects_csv(
                f, fieldnames=['', 'xmin', 'ymin', 'xmax', 'ymax'], init_time=1, delim=' ')

    # Take a subset of the labels.
    if fps != 240:
        if 240 % fps != 0:
            raise RuntimeError('fps does not divide 240: {}'.format(fps))
        freq = 240 // fps
        labels_pix = {v: _subsample(labels_pix[v], freq) for v in video_ids}

    image_file_fn = functools.partial(_image_file, fps=fps)
    labels, aspects = dataset.convert_relative(dir, video_ids, labels_pix, image_file_fn)
    return dataset.Dataset(
        track_ids=video_ids, labels=labels,
        image_files=util.func_dict(video_ids, image_file_fn),
        aspects=aspects)


def _subsample(labels, freq):
    times = labels.keys()
    min_time, max_time = min(times), max(times)
    if min_time != 1:
        raise RuntimeError('first time is not 1: {}'.format(min_time))
    return {t // freq + 1: labels[t + 1] for t in range(0, max_time, freq) if t + 1 in labels}


def _discover_tracks(dir, fps):
    subdirs = util.list_subdirs(dir)
    # Check if annotation file exists.
    return [v for v in subdirs if os.path.isfile(os.path.join(dir, _annot_file(v, fps)))]


def _annot_file(video_id, fps):
    return os.path.join(video_id, str(fps), video_id + '.txt')


def _image_file(video_id, fps):
    return os.path.join(video_id, str(fps), video_id, '{:05d}.jpg')
