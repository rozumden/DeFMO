import os

## TODO: insert your dataset folder and folder for storing results
dataset_folder = '/cluster/scratch/denysr/dataset/'
tmp_folder = '/cluster/home/denysr/tmp/'
run_folder = tmp_folder+'PyTorch/'

import datetime
strdate = datetime.datetime.now().strftime("%Y%m%d_%H%M")

g_tbd_folder = dataset_folder+'TbD_GC/'
g_tbd3d_folder = dataset_folder+'TbD-3D/'
g_falling_folder = dataset_folder+'falling_objects/'
g_wildfmo_folder = dataset_folder+'wildfmo/'
g_youtube_folder = dataset_folder+'youtube/'

## TODO: provide valid training and testing/validation dataset paths
g_dataset_folder = dataset_folder+'ShapeNetv2/ShapeBlur1000STL.hdf5'
g_validation_folder = dataset_folder+'ShapeNetv2/ShapeBlur20STL.hdf5'
g_temp_folder = run_folder+strdate+'_defmotest'

g_number_per_category = 1000

g_render_objs_all = ['table','jar', 'skateboard', 'bottle' , 'tower' ,'chair' ,'bookshelf' ,'camera' ,'laptop' ,'basket' , 'sofa' ,'knife' , 'can' , 'rifle' , 'train' ,  'lamp' ,  'trash bin' , 'mailbox' , 'watercraft' , 'motorbike' , 'dishwasher' ,  'bench' , 'pistol' , 'rocket' , 'loudspeaker' , 'file cabinet' ,  'bag' , 'cabinet' , 'bed' , 'birdhouse' , 'display' , 'piano' , 'earphone' , 'telephone' , 'stove' , 'microphone', 'mug', 'remote', 'bathtub' ,  'bowl' , 'keyboard', 'guitar' , 'washer', 'faucet' , 'printer' , 'cap' , 'clock', 'helmet', 'flowerpot', 'microwaves']
g_render_objs_test = ['pillow']
g_render_objs_train = g_render_objs_all
g_render_objs = g_render_objs_train

g_number_per_category_val = 20
# g_render_objs_val = ['bottle', 'bowl', 'can', 'jar', 'mug']
g_render_objs_val = g_render_objs_all

g_epochs = 30
g_batch_size = 7*3
g_resolution_x = int(640/2)
g_resolution_y = int(480/2)
g_fmo_steps = 24
g_fmo_train_steps = 2*12 # must be even
g_use_median = True
g_normalize = False

g_sharp_mask_type = 'entropy'
g_timeconsistency_type = 'ncc' # oflow, ncc

g_use_selfsupervised_model = True
g_use_selfsupervised_sharp_mask = True
g_use_selfsupervised_timeconsistency = True
g_use_supervised = True
g_use_latent_learning = False

g_finetune = False

g_lr = 1e-3

## for speed-up and memory
g_num_workers = 6


g_eval_d = 5