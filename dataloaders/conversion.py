import argparse
import cv2
import h5py
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

import os
dataset_folder = '/mnt/lascar/rozumden/dataset/'
tmp_folder = '/home.stud/rozumden/tmp/'
if not os.path.exists(dataset_folder):
    dataset_folder = '/cluster/scratch/denysr/dataset/'
    tmp_folder = '/cluster/home/denysr/tmp/'
g_dataset_folder = dataset_folder+'ShapeNetv2/ShapeBlur1000STL'

import pdb

def main():
    parser = argparse.ArgumentParser(
        description='Load the images, convert them to the network input and '
                    'store it as a dataset')
    parser.add_argument('-i', '--input-dir', default=g_dataset_folder,
                        help='Input directory containing the image pairs')
    parser.add_argument('-o', '--output-file', default=g_dataset_folder+'.hdf5',
                        help='Output dataset file')
    parser.add_argument('-c', dest='compress', action='store_true',
                        help='Compress the data with lzf compression')

    args = parser.parse_args()

    if args.compress:
        compression_type = "lzf"
    else:
        compression_type = None

    all_filenames = [f for f in os.listdir(args.input_dir)]

    # create h5 file where all the data wll be stored
    if os.path.exists(args.output_file):
        # TODO ask for confirmation to delete old file
        os.remove(args.output_file)
    print(os.getcwd())
    output_file = h5py.File(args.output_file)

    for obj_type in tqdm(all_filenames):
        # load best aligned images

        output_file.create_group(obj_type)
        all_objects = [f for f in os.listdir(os.path.join(args.input_dir, obj_type)) if '.png' in f]

        for obj_name in tqdm(all_objects):
            name = '_'.join(map(str, obj_name.split('.')[:-1]))
            output_file[obj_type].create_group(name)

            im_path = str(os.path.join(args.input_dir, obj_type, obj_name))
            bgr_path = str(os.path.join(args.input_dir, obj_type, "GT", name, "bgr.png"))
            bgrmed_path = str(os.path.join(args.input_dir, obj_type, "GT", name, "bgr_med.png"))

            # load the images
            im = cv2.imread(str(im_path), -1)
            bgr = cv2.imread(str(bgr_path), -1)
            bgr_med = cv2.imread(str(bgrmed_path), -1)

            # adding data to dataset file
            output_file[obj_type][name].create_dataset('im', data=im, compression=compression_type)
            output_file[obj_type][name].create_dataset('bgr', data=bgr, compression=compression_type)
            output_file[obj_type][name].create_dataset('bgr_med', data=bgr_med, compression=compression_type)

            all_gt = [f for f in os.listdir(os.path.join(args.input_dir, obj_type, "GT", name)) if 'image' in f]
            output_file[obj_type][name].create_group("GT")
            for gtname in tqdm(all_gt):
                gtsavename = '_'.join(map(str, gtname.split('.')[:-1]))
                gtim = cv2.imread(str(os.path.join(args.input_dir, obj_type, "GT", name, gtname)), -1)
                output_file[obj_type][name]["GT"].create_dataset(gtsavename, data=gtim, compression=compression_type)


    output_file.close()

if __name__ == "__main__":
    main()
