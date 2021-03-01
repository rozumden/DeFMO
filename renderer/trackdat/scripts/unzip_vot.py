from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='e.g. dl/vot2018')
    parser.add_argument('dl_dir', help='e.g. data/vot2018')
    args = parser.parse_args()

    with open(os.path.join(args.data_dir, 'description.json'), 'r') as f:
        dataset = json.load(f)

    for seq in dataset['sequences']:
        annotations_zip = os.path.join(args.data_dir, 'annotations',
                                       http_basename(seq['annotations']['url']))
        color_zip = os.path.join(args.data_dir, 'color',
                                 http_basename(seq['channels']['color']['url']))
        dst_dir = os.path.join(args.dl_dir, seq['name'])
        extract(annotations_zip, dst_dir)
        extract(color_zip, dst_dir)


def extract(filename, dir=None):
    print('extract "{}"'.format(filename))
    with zipfile.ZipFile(filename, 'r') as zf:
        zf.extractall(dir)


def http_basename(s):
    return s.split('/')[-1]


if __name__ == '__main__':
    main()
