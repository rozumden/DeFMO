'''

Expects directory structure:
    data_seq/UAV123/{video}/{frame:06d}.jpg
    anno/subset/{video}.txt
    anno/subset/{video}_{object_num}.txt
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import fnmatch
import functools
import os
import re

from . import dataset
from . import util


def load_uav123(dir, subset='UAV123'):
    # Cannot use load_csv_dataset_simple because
    # there are multiple tracks in some videos.

    track_ids = _load_tracks_subset(dir, subset)
    init_times = _init_times_subset(subset)

    labels_pix = {}
    for track_id in track_ids:
        with open(os.path.join(dir, _annot_file(subset, track_id)), 'r') as f:
            labels_pix[track_id] = dataset.load_rects_csv(
                f, fieldnames=['xmin', 'ymin', 'width', 'height'],
                init_time=init_times.get(track_id, 1),
                delim=',')

    video_id_map = util.func_dict(track_ids, _video_id_from_track_id)
    video_ids = set(video_id_map.values())
    labels, aspects = dataset.convert_relative(
        dir, track_ids, labels_pix, _image_file, video_id_map)
    return dataset.Dataset(
        track_ids=track_ids, labels=labels, video_id_map=video_id_map,
        image_files=util.func_dict(video_ids, _image_file),
        aspects=aspects)


def _load_tracks_subset(dir, subset):
    elems = os.listdir(os.path.join(dir, 'UAV123', 'anno', subset))
    annot_files = fnmatch.filter(elems, '*.txt')
    return list(map(_remove_ext, annot_files))


def _annot_file(subset, track_id):
    return os.path.join('UAV123', 'anno', subset, track_id + '.txt')


def _image_file(video_id):
    return os.path.join('UAV123', 'data_seq', 'UAV123', video_id, '{:06d}.jpg')


def _remove_ext(name):
    root, _ = os.path.splitext(name)
    return root


def _video_id_from_track_id(track_id):
    # Remove underscore and numbers.
    return re.sub('_[\d]+$', '', track_id)


def _init_times_subset(subset):
    if subset == 'UAV123':
        return INIT_TIMES_UAV123
    elif subset == 'UAV20L':
        return INIT_TIMES_UAV20L
    else:
        raise RuntimeError('unknown subset: {}'.format(subset))


# From UAV123/configSeqs.m
INIT_TIMES_UAV20L = {
    # 'bike1': 1,
    # 'bird1': 1,
    # 'car1': 1,
    # 'car3': 1,
    # 'car6': 1,
    # 'car8': 1,
    # 'car9': 1,
    # 'car16': 1,
    # 'group1': 1,
    # 'group2': 1,
    # 'group3': 1,
    # 'person2': 1,
    # 'person4': 1,
    # 'person5': 1,
    # 'person7': 1,
    # 'person14': 1,
    # 'person17': 1,
    # 'person19': 1,
    # 'person20': 1,
    # 'uav1': 1,
}

# From UAV123/configSeqs.m
INIT_TIMES_UAV123 = {
    # 'bike1': 1,
    # 'bike2': 1,
    # 'bike3': 1,
    # 'bird1_1': 1,
    'bird1_2': 775,
    'bird1_3': 1573,
    # 'boat1': 1,
    # 'boat2': 1,
    # 'boat3': 1,
    # 'boat4': 1,
    # 'boat5': 1,
    # 'boat6': 1,
    # 'boat7': 1,
    # 'boat8': 1,
    # 'boat9': 1,
    # 'building1': 1,
    # 'building2': 1,
    # 'building3': 1,
    # 'building4': 1,
    # 'building5': 1,
    # 'car1_1': 1,
    'car1_2': 751,
    'car1_3': 1627,
    # 'car2': 1,
    # 'car3': 1,
    # 'car4': 1,
    # 'car5': 1,
    # 'car6_1': 1,
    'car6_2': 487,
    'car6_3': 1807,
    'car6_4': 2953,
    'car6_5': 3925,
    # 'car7': 1,
    # 'car8_1': 1,
    'car8_2': 1357,
    # 'car9': 1,
    # 'car10': 1,
    # 'car11': 1,
    # 'car12': 1,
    # 'car13': 1,
    # 'car14': 1,
    # 'car15': 1,
    # 'car16_1': 1,
    'car16_2': 415,
    # 'car17': 1,
    # 'car18': 1,
    # 'group1_1': 1,
    'group1_2': 1333,
    'group1_3': 2515,
    'group1_4': 3925,
    # 'group2_1': 1,
    'group2_2': 907,
    'group2_3': 1771,
    # 'group3_1': 1,
    'group3_2': 1567,
    'group3_3': 2827,
    'group3_4': 4369,
    # 'person1': 1,
    # 'person2_1': 1,
    'person2_2': 1189,
    # 'person3': 1,
    # 'person4_1': 1,
    'person4_2': 1501,
    # 'person5_1': 1,
    'person5_2': 877,
    # 'person6': 1,
    # 'person7_1': 1,
    'person7_2': 1249,
    # 'person8_1': 1,
    'person8_2': 1075,
    # 'person9': 1,
    # 'person10': 1,
    # 'person11': 1,
    # 'person12_1': 1,
    'person12_2': 601,
    # 'person13': 1,
    # 'person14_1': 1,
    'person14_2': 847,
    'person14_3': 1813,
    # 'person15': 1,
    # 'person16': 1,
    # 'person17_1': 1,
    'person17_2': 1501,
    # 'person18': 1,
    # 'person19_1': 1,
    'person19_2': 1243,
    'person19_3': 2791,
    # 'person20': 1,
    # 'person21': 1,
    # 'person22': 1,
    # 'person23': 1,
    # 'truck1': 1,
    # 'truck2': 1,
    # 'truck3': 1,
    # 'truck4_1': 1,
    'truck4_2': 577,
    # 'uav1_1': 1,
    'uav1_2': 1555,
    'uav1_3': 2473,
    # 'uav2': 1,
    # 'uav3': 1,
    # 'uav4': 1,
    # 'uav5': 1,
    # 'uav6': 1,
    # 'uav7': 1,
    # 'uav8': 1,
    # 'wakeboard1': 1,
    # 'wakeboard2': 1,
    # 'wakeboard3': 1,
    # 'wakeboard4': 1,
    # 'wakeboard5': 1,
    # 'wakeboard6': 1,
    # 'wakeboard7': 1,
    # 'wakeboard8': 1,
    # 'wakeboard9': 1,
    # 'wakeboard10': 1,
    # 'car1_s': 1,
    # 'car2_s': 1,
    # 'car3_s': 1,
    # 'car4_s': 1,
    # 'person1_s': 1,
    # 'person2_s': 1,
    # 'person3_s': 1,
}
