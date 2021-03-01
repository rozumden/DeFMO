'''

Expects directory structure:
    {set}/labels_frames.csv
    {set}/frames/{video}/{time_sec:05d}.jpg
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os

from . import dataset
from . import util

_FIELDNAMES = [
    'youtube_id', 'timestamp_ms', 'class_id', 'class_name', 'object_id',
    'object_presence', 'xmin', 'xmax', 'ymin', 'ymax',
]


def load_ytbb_sec(dir, subset, no_aspect=False, keep_pure_absent=False):
    '''Loads YTBB dataset with one frame per second.'''
    # csv_file = os.path.join(dir, 'yt_bb_detection_{}.csv'.format(subset))
    csv_file = os.path.join(dir, subset, 'labels_frames.csv')
    video_id_map = {}
    labels = {}
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f, fieldnames=_FIELDNAMES)
        for row in reader:
            track_id = '{}_{}_{}'.format(row['youtube_id'], row['class_id'], row['object_id'])
            time_ms = int(row['timestamp_ms'])
            if time_ms % 1000 != 0:
                raise RuntimeError('timestamp not divisible by 1000: {}'.format(time_ms))
            t = time_ms // 1000
            if row['object_presence'] == 'present':
                frame_label = dataset.make_frame_label(
                    rect=dataset.make_rect(
                        xmin=float(row['xmin']), xmax=float(row['xmax']),
                        ymin=float(row['ymin']), ymax=float(row['ymax'])))
            elif row['object_presence'] == 'absent':
                frame_label = dataset.make_frame_label(absent=True)
            else:
                continue
            video_id_map[track_id] = row['youtube_id']
            labels.setdefault(track_id, {})[t] = frame_label
    if not keep_pure_absent:
        num_raw = len(labels)
        labels = {track_id: track_labels for track_id, track_labels in labels.items()
                  if _num_present(track_labels) > 1}
        print('remove tracks without present labels: {} of {} remain'.format(len(labels), num_raw))
    track_ids = list(labels.keys())
    video_id_map = {track_id: video_id_map[track_id] for track_id in track_ids}
    video_ids = set(video_id_map[track_id] for track_id in track_ids)
    # print('num videos:', len(video_ids))

    # # Check if videos exist.
    # video_subset = set([video_id for video_id in video_ids
    #                     if os.path.isdir(os.path.join(dir, subset, 'frames', video_id))])
    # track_subset = set([track_id for track_id in track_ids
    #                     if video_id_map[track_id] in video_subset])
    # print('found frames for {} of {} videos ({} of {} tracks)'.format(
    #     len(video_subset), len(video_ids), len(track_subset), len(track_ids)))
    # # Take subset of tracks for videos that exist.
    # video_ids = video_subset
    # track_ids = [track_id for track_id in track_ids if track_id in track_subset]
    # labels = {track_id: labels[track_id] for track_id in track_subset}
    # video_id_map = {track_id: video_id_map[track_id] for track_id in track_subset}

    def image_file_fn(v): return _image_file(subset, v)
    # Rectangles are already in relative coordinates.
    # However we must read images to get aspect ratios.
    if no_aspect:
        aspects = None
    else:
        aspects = dataset.aspect_from_images(dir, track_ids, labels, image_file_fn, video_id_map)

    return dataset.Dataset(
        track_ids=track_ids, labels=labels, video_id_map=video_id_map,
        image_files=util.func_dict(video_ids, image_file_fn),
        aspects=aspects)


def _image_file(subset, video_id):
    return os.path.join(subset, 'frames', video_id, '{:05d}.jpg')


def _num_present(labels):
    return sum(1 for label in labels.values() if dataset.is_present(label))
