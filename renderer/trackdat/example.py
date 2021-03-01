from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import time

import trackdat
import trackdat.dataset

alov_dir = 'alov'
dtb70_dir = 'dtb70'
ilsvrc_dir = 'ilsvrc'
nfs_dir = 'nfs'
nuspro_dir = 'nuspro'
otb_dir = 'otb'
tc128_dir = 'tc128'
tlp_dir = 'tlp'
uav123_dir = 'uav123'
vot2013_dir = 'vot2013'
vot2014_dir = 'vot2014'
vot2015_dir = 'vot2015'
vot2016_dir = 'vot2016'
vot2017_dir = 'vot2017'
ytbb_dir = 'ytbb'

datasets = [
    dict(func=trackdat.load_alov, dir=alov_dir, kwargs=dict()),
    dict(func=trackdat.load_dtb70, dir=dtb70_dir, kwargs=dict()),
    dict(func=trackdat.load_ilsvrc, dir=ilsvrc_dir, kwargs=dict(subset='val')),
    dict(func=trackdat.load_ilsvrc, dir=ilsvrc_dir, kwargs=dict(subset='train')),
    dict(func=trackdat.load_nfs, dir=nfs_dir, kwargs=dict(fps=240)),
    dict(func=trackdat.load_nfs, dir=nfs_dir, kwargs=dict(fps=30)),
    dict(func=trackdat.load_nuspro, dir=nuspro_dir, kwargs=dict()),
    dict(func=trackdat.load_otb, dir=otb_dir, kwargs=dict()),
    dict(func=trackdat.load_otb, dir=otb_dir, kwargs=dict(subset='cvpr13')),
    dict(func=trackdat.load_otb, dir=otb_dir, kwargs=dict(subset='tb_50')),
    dict(func=trackdat.load_otb, dir=otb_dir, kwargs=dict(subset='tb_100')),
    dict(func=trackdat.load_tc128, dir=tc128_dir, kwargs=dict(keep_prev=False)),
    dict(func=trackdat.load_tc128, dir=tc128_dir, kwargs=dict(keep_prev=True)),
    dict(func=trackdat.load_tlp, dir=tlp_dir, kwargs=dict()),
    dict(func=trackdat.load_uav123, dir=uav123_dir, kwargs=dict()),
    dict(func=trackdat.load_vot, dir=vot2013_dir, kwargs=dict()),
    dict(func=trackdat.load_vot, dir=vot2014_dir, kwargs=dict()),
    dict(func=trackdat.load_vot, dir=vot2015_dir, kwargs=dict()),
    dict(func=trackdat.load_vot, dir=vot2016_dir, kwargs=dict()),
    dict(func=trackdat.load_vot, dir=vot2017_dir, kwargs=dict()),
    dict(func=trackdat.load_ytbb_sec, dir=ytbb_dir, kwargs=dict(subset='validation')),
    dict(func=trackdat.load_ytbb_sec, dir=ytbb_dir, kwargs=dict(subset='train')),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_root', metavar='data/')
    args = parser.parse_args()

    for x in datasets:
        print(x)
        dataset_dir = os.path.join(args.data_root, x['dir'])
        start = time.time()
        dataset = x['func'](dataset_dir, **x['kwargs'])
        dur = time.time() - start
        print('number of tracks:', len(dataset.tracks()))
        print('time to load: {:.3g} sec'.format(dur))
        trackdat.dataset.assert_image_files_exist(dataset_dir, dataset)


if __name__ == '__main__':
    main()
