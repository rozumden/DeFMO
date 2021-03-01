from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math
import os
import re

from . import util

logger = logging.getLogger(__name__)


def make_rect(xmin, ymin, xmax, ymax):
    rect = dict(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
    _assert_rect_ok(rect)
    return rect


def make_rect_pix(xmin, ymin, xmax, ymax, imwidth, imheight):
    return make_rect(
        xmin=float(xmin) / imwidth,
        xmax=float(xmax) / imwidth,
        ymin=float(ymin) / imheight,
        ymax=float(ymax) / imheight)


def make_frame_label(rect=None, absent=False, extra=None):
    frame = {}
    # Interpret empty/reversed rectangle as absent.
    if rect is not None and not is_non_empty(rect):
        rect = None
        absent = True
    if rect is not None:
        frame['rect'] = rect
    if absent:
        frame['absent'] = True
    if extra:
        frame['extra'] = extra
    return frame


def is_present(label):
    return not label.get('absent', False)


def is_non_empty(rect):
    return rect['xmin'] < rect['xmax'] and rect['ymin'] < rect['ymax']


def label_convert_relative(frame_label, im_size):
    frame_label = dict(frame_label)
    if 'rect' in frame_label:
        kwargs = dict(frame_label['rect'])
        kwargs['imwidth'] = im_size['width']
        kwargs['imheight'] = im_size['height']
        frame_label['rect'] = make_rect_pix(**kwargs)
    else:
        if not frame_label.get('absent', False):
            raise RuntimeError('no rectangle and not absent: ' + str(frame_label))
    return frame_label


class Dataset(object):

    def __init__(self, track_ids, labels, image_files, aspects, video_id_map=None):
        '''
        Args:
            track_ids: List of `track_id` strings.
            labels: Dict that maps `track_id` to label dict.
            image_files: Dict that maps `video_id` to format string.
            aspects: Dict that maps `video_id` to float (width / height).
            video_id_map: Dict that maps `track_id` to `video_id`.
                If `track_id` is not found, then `track_id` is used as `video_id`.
        '''
        self._track_ids = track_ids
        self._labels = labels
        self._video_id_map = video_id_map or {}
        self._image_files = image_files
        self._aspects = aspects

    def tracks(self):
        return self._track_ids

    def video(self, track_id):
        # Default to track ID itself.
        return self._video_id_map.get(track_id, track_id)

    def labels(self, track_id):
        return self._labels[track_id]

    def image_file(self, video_id, time):
        return self._image_files[video_id].format(time)

    def aspect(self, video_id):
        return self._aspects[video_id]


def to_dict(dataset):
    fields = ['track_ids', 'labels', 'video_id_map', 'image_files', 'aspects']
    content = {field: dataset.__dict__['_' + field] for field in fields}
    # JSON does not support map with integer keys.
    # Convert to list of (integer, value) pairs instead.
    content['labels'] = {track_id: sorted(track.items())
                         for track_id, track in content['labels'].items()}
    return content


def from_dict(content):
    content = dict(content)
    # Convert from list of [t, dict] pairs.
    content['labels'] = util.map_dict_values(dict, content['labels'])
    return Dataset(**content)


def load_csv_dataset_simple(dir, load_videos_fn, annot_file_fn, image_file_fn,
                            fieldnames=None, init_time=None, delim=','):
    '''Load simple dataset (where each video has one track).'''
    video_ids = load_videos_fn(dir)
    if len(video_ids) == 0:
        raise RuntimeError('no tracks found')

    labels_pix = {}
    for video_id in video_ids:
        annot_file = os.path.join(dir, annot_file_fn(video_id))
        try:
            with open(annot_file, 'r') as f:
                labels_pix[video_id] = load_rects_csv(f,
                                                      fieldnames=fieldnames,
                                                      init_time=init_time,
                                                      delim=delim)
        except (KeyboardInterrupt, SystemExit):
            raise
        except RuntimeError as ex:
            raise RuntimeError('error loading rects from file "{}": {}'.format(annot_file, ex))
    labels, aspects = convert_relative(dir, video_ids, labels_pix, image_file_fn)
    return Dataset(
        track_ids=video_ids, labels=labels,
        image_files=util.func_dict(video_ids, image_file_fn),
        aspects=aspects)


def load_rects_csv(f, fieldnames, init_time=None, delim=','):
    '''Loads rectangles from a CSV file.

    Does not perform any conversion of units.

    If 'time' in fieldnames, then it uses the time column.
    Otherwise, time starts at init_time and proceeds in increments of 1.
    '''
    re_delim = re.compile(delim)
    time_is_field = 'time' in fieldnames
    absent_is_field = 'absent' in fieldnames
    if 'xmin' in fieldnames:
        if 'xmax' in fieldnames:
            label_fn = _rect_min_max
        else:
            label_fn = _rect_min_size
    elif 'x0' in fieldnames:
        label_fn = _rect_corners
    else:
        raise RuntimeError('unknown fields: {}'.format(', '.join(fieldnames)))

    if not time_is_field and init_time is None:
        raise RuntimeError('must specify init time if time is not a field')
    t = init_time
    labels = {}
    for line in f:
        fields = re_delim.split(line.strip())
        row = {k: v for k, v in zip(fieldnames, fields) if k}
        if time_is_field:
            t = int(row['time'])
        if absent_is_field and _flexibool(row['absent']):
            labels[t] = make_frame_label(absent=True)
        else:
            labels[t] = label_fn(row)
        # if is_present(labels[t]):
        #     assert is_non_empty(labels[t]['rect']), \
        #         'bad rect from line \'{}\' in file \'{}\''.format(line.strip(), f.name)
        t += 1
    return labels


_TRUE_VALUES = {'1', 'true', 't', 'yes', 'y'}
_FALSE_VALUES = {'0', 'false', 'f', 'no', 'n'}


def _flexibool(s):
    s = s.strip()
    s = s.lower()
    if s in _TRUE_VALUES:
        return True
    elif s in _FALSE_VALUES:
        return False
    else:
        raise ValueError('unsupported boolean value: {}'.format(s))


def _rect_min_size(row):
    xmin = float(row['xmin'])
    ymin = float(row['ymin'])
    width = float(row['width'])
    height = float(row['height'])
    if any(map(math.isnan, [xmin, ymin, width, height])):
        return make_frame_label(absent=True)
    return make_frame_label(rect=make_rect(xmin=xmin, ymin=ymin,
                                           xmax=xmin + width, ymax=ymin + height))


def _rect_min_max(row):
    xmin = float(row['xmin'])
    ymin = float(row['ymin'])
    xmax = float(row['xmax'])
    ymax = float(row['ymax'])
    if any(map(math.isnan, [xmin, ymin, xmax, ymax])):
        return make_frame_label(absent=True)
    return make_frame_label(rect=make_rect(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax))


def _rect_corners(row):
    xs = [float(row[key]) for key in ['x0', 'x1', 'x2', 'x3']]
    ys = [float(row[key]) for key in ['y0', 'y1', 'y2', 'y3']]
    if any(map(math.isnan, xs + ys)):
        return make_frame_label(absent=True)
    if len(set(xs)) != 2:
        raise RuntimeError('not 2 unique x values: {}'.format(str(xs)))
    if len(set(ys)) != 2:
        raise RuntimeError('not 2 unique y values: {}'.format(str(ys)))
    return make_frame_label(rect=make_rect(xmin=min(xs), ymin=min(ys), xmax=max(xs), ymax=max(ys)))


def _assert_rect_ok(rect):
    # If either is NaN, then comparison will fail.
    if not rect['xmin'] <= rect['xmax']:
        raise RuntimeError('expect xmin <= xmax: {}, {}'.format(rect['xmin'], rect['xmax']))
    if not rect['ymin'] <= rect['ymax']:
        raise RuntimeError('expect ymin <= ymax: {}, {}'.format(rect['ymin'], rect['ymax']))


def _flip_rect_if_necessary(rect):
    rect = dict(rect)
    if rect['xmax'] < rect['xmin']:
        logger.warning('flip inverted rect: xmin %g, xmax %g', rect['xmin'], rect['xmax'])
        rect['xmin'], rect['xmax'] = rect['xmax'], rect['xmin']
    if rect['ymax'] < rect['ymin']:
        logger.warning('flip inverted rect: ymin %g, ymax %g', rect['ymin'], rect['ymax'])
        rect['ymin'], rect['ymax'] = rect['ymax'], rect['ymin']
    return rect


def assert_image_files_exist(dir, dataset):
    for track_id in dataset.tracks():
        video_id = dataset.video(track_id)
        times = dataset.labels(track_id).keys()
        # Check first and last times.
        util.assert_file_exists(os.path.join(dir, dataset.image_file(video_id, min(times))))
        util.assert_file_exists(os.path.join(dir, dataset.image_file(video_id, max(times))))


def convert_relative(dir, track_ids, labels_pix, image_file_fn, video_id_map=None):
    '''Converts labels from pixels to relative coords using image size.'''
    video_id_map = video_id_map or {}

    im_sizes = {}
    for track_id in track_ids:
        video_id = video_id_map.get(track_id, track_id)
        if video_id in im_sizes:
            # Already have size information for this video (multiple tracks per video).
            continue
        first_image = image_file_fn(video_id).format(min(labels_pix[track_id].keys()))
        im_sizes[video_id] = util.image_size(os.path.join(dir, first_image))

    labels = {}
    for track_id in track_ids:
        video_id = video_id_map.get(track_id, track_id)
        labels[track_id] = util.map_dict_values(
            lambda x: label_convert_relative(x, im_sizes[video_id]),
            labels_pix[track_id])
    aspects = {video_id: float(size['width']) / size['height']
               for video_id, size in im_sizes.items()}
    return labels, aspects


def aspect_from_images(dir, track_ids, labels_pix, image_file_fn, video_id_map=None):
    video_id_map = video_id_map or {}

    im_sizes = {}
    for track_id in track_ids:
        video_id = video_id_map.get(track_id, track_id)
        if video_id in im_sizes:
            # Already have size information for this video (multiple tracks per video).
            continue
        first_image = image_file_fn(video_id).format(min(labels_pix[track_id].keys()))
        im_sizes[video_id] = util.image_size(os.path.join(dir, first_image))

    return {video_id: float(size['width']) / size['height']
            for video_id, size in im_sizes.items()}
