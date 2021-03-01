'''
Expects directory structure:
    {chunk}/anno/{video}.txt
    {chunk}/frames/{video}/{frame:d}.jpg
'''

# Based on:
# https://github.com/SilvioGiancola/TrackingNet-devkit/blob/master/generate_BB_frames.py

import os
import fnmatch
import functools
import itertools

from . import dataset
from . import util


_SUBSET_CHUNKS = {
    'train': ['TRAIN_{:d}'.format(i) for i in range(12)],
    'test': ['TEST'],
}

_ANNOT_FIELDNAMES = ['xmin', 'ymin', 'width', 'height']


def load_trackingnet(dir, subset, rate=1):
    '''
    Args:
        subset: {'train', 'test'}
    '''
    ds = dataset.load_csv_dataset_simple(
        dir,
        load_videos_fn=functools.partial(_load_videos, subset=subset),
        annot_file_fn=_annot_file,
        image_file_fn=_image_file,
        fieldnames=_ANNOT_FIELDNAMES,
        init_time=0,
        delim=',')

    if rate and rate > 1:
        ds._labels = {
            track_id: {
                t: label for t, label in track_labels.items() if t % rate == 0
            } for track_id, track_labels in ds._labels.items()
        }

    return ds


def _load_videos(dir, subset):
    chunks = _SUBSET_CHUNKS[subset]
    return list(itertools.chain.from_iterable(_get_chunk_videos(dir, chunk) for chunk in chunks))


def _annot_file(video_id):
    chunk, basename = video_id.split('/')
    return os.path.join(chunk, 'anno', basename + '.txt')


def _image_file(video_id):
    chunk, basename = video_id.split('/')
    return os.path.join(chunk, 'frames', basename, '{:d}.jpg')


def _get_chunk_videos(dataset_dir, chunk):
    annot_dir = os.path.join(dataset_dir, chunk, 'anno')
    elems = os.listdir(annot_dir)
    annot_files = fnmatch.filter(elems, '*.txt')
    if len(annot_files) == 0:
        raise RuntimeError('no annotations found in {}'.format(annot_dir))
    basenames = sorted(map(_remove_ext, annot_files))
    video_ids = [chunk + '/' + basename for basename in basenames]
    return video_ids


def _remove_ext(name):
    root, _ = os.path.splitext(name)
    return root
