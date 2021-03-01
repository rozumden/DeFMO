'''

Expects directory structure:
    Data/VID/{subset}/{video}/{frame:06d}.JPEG
    Annotations/VID/{subset}/{video}/{frame}.xml
    ImageSets/VID/train_{class_num}.txt
    ImageSets/VID/val.txt
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import fnmatch
import itertools
import os

from xml.etree import ElementTree as etree

from . import util
from . import dataset


def load_ilsvrc(dir, subset):
    '''
    Args:
        convert_args: See `convert_track_annotation`
    '''
    if subset == 'train':
        snippets = _load_snippets_train(dir)
    elif subset == 'val':
        snippets = _load_snippets_val(dir)
    else:
        raise ValueError('unknown set: {}'.format(subset))

    track_ids = []
    labels = {}
    aspects = {}
    video_id_map = {}
    for snippet_id in snippets:
        snippet_labels, im_size = _load_snippet_labels(dir, subset, snippet_id)
        labels.update(snippet_labels)
        aspects[snippet_id] = float(im_size['width']) / im_size['height']
        snippet_track_ids = list(sorted(snippet_labels.keys()))
        track_ids.extend(snippet_track_ids)
        video_id_map.update({track_id: snippet_id for track_id in snippet_track_ids})

    return dataset.Dataset(
        track_ids=track_ids, labels=labels, video_id_map=video_id_map,
        image_files=util.func_dict(snippets, lambda v: _image_file(subset, v)),
        aspects=aspects)


def _image_file(subset, snippet_id):
    parts = ['ILSVRC2015', 'Data', 'VID', subset] + snippet_id.split('/') + ['{:06d}.JPEG']
    return os.path.join(*parts)


def _load_snippets_train(dir, subset='train', num_classes=30):
    # Load training snippets for each class.
    # For the train set, there is a file per class that lists positive and negative snippets.
    class_snippets = [_load_positive_snippets(dir, subset, num)
                      for num in range(1, num_classes + 1)]
    # Take union of all sets.
    snippets = set(itertools.chain.from_iterable(class_snippets))
    return snippets


def _load_positive_snippets(dir, subset, class_num):
    set_file = '{}_{:d}.txt'.format(subset, class_num)
    path = os.path.join(dir, 'ILSVRC2015', 'ImageSets', 'VID', set_file)
    with open(path, 'r') as f:
        reader = csv.DictReader(f, delimiter=' ', fieldnames=['snippet_id', 'label'])
        rows = list(reader)
    snippets = [r['snippet_id'] for r in rows if r['label'] == '1']
    return snippets


def _load_snippets_val(dir, subset='val'):
    # For the val set, there is a file val.txt that lists all frames of all videos.
    path = os.path.join(dir, 'ILSVRC2015', 'ImageSets', 'VID', subset + '.txt')
    with open(path, 'r') as f:
        reader = csv.DictReader(f, delimiter=' ', fieldnames=['frame_name', 'frame_index'])
        rows = list(reader)
    # Take unique snippet names.
    snippets = set(_snippet_from_frame_name(r['frame_name']) for r in rows)
    return snippets


def _make_track_id(snippet_id, object_id):
    return '_'.join([snippet_id, 'object', object_id])


def _load_snippet_labels(dir, subset, snippet_id):
    n = _snippet_length(dir, subset, snippet_id)
    dir_name = os.path.join(dir, 'ILSVRC2015', 'Annotations', 'VID', subset, snippet_id)

    labels = {}
    for t in range(n):
        annot_file = os.path.join(dir_name, '{:06d}.xml'.format(t))
        tree = etree.parse(annot_file)
        annot = tree.getroot()
        if annot.tag != 'annotation':
            raise RuntimeError('root tag is not annotation: {}'.format(annot.tag))
        size = annot.find('size')
        if size is None:
            raise RuntimeError('no size tag')
        imwidth = int(size.find('width').text.strip())
        imheight = int(size.find('height').text.strip())
        for obj in annot.findall('object'):
            object_id = obj.find('trackid').text
            track_id = _make_track_id(snippet_id, object_id)
            labels.setdefault(track_id, {})[t] = _label_from_obj_node(
                obj, imwidth=imwidth, imheight=imheight)

    # Fill in missing times with 'absent' annotations.
    for t in range(n):
        # absent = [track_id for track_id in labels.keys() if t not in labels[track_id]]
        for track_id in labels.keys():
            if t not in labels[track_id]:
                labels[track_id][t] = dataset.make_frame_label(absent=True)

    return labels, {'width': imwidth, 'height': imheight}


def _snippet_length(dir, subset, snippet_id):
    dir_name = os.path.join(dir, 'ILSVRC2015', 'Data', 'VID', subset, snippet_id)
    image_files = fnmatch.filter(os.listdir(dir_name), '*.JPEG')
    return len(image_files)


def _snippet_from_frame_name(s):
    # Takes frame name from val.txt and gets snippet name.
    parts = s.split('/')
    return '/'.join(parts[:-1])


def _label_from_obj_node(obj, imwidth, imheight):
    extra = {}
    for field in ['occluded', 'generated']:
        field_node = obj.find(field)
        if field_node is None:
            extra[field] = field_node.text

    bndbox = obj.find('bndbox')
    xmin = float(bndbox.find('xmin').text)
    xmax = float(bndbox.find('xmax').text)
    ymin = float(bndbox.find('ymin').text)
    ymax = float(bndbox.find('ymax').text)
    rect = dataset.make_rect_pix(
        xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax,
        imwidth=imwidth, imheight=imheight)
    return dataset.make_frame_label(rect=rect, extra=extra)
