import h5py
import os, glob
import numpy as np

from main_settings import *
from renderer.settings import *
from utils import *
from PIL import Image
import random
import cv2
from skimage.measure import regionprops

import pdb

def renderTraj(pars, H):
    ## Input: pars is either 2x2 (line) or 2x3 (parabola)
    if pars.shape[1] == 2:
        pars = np.concatenate( (pars, np.zeros((2,1))),1)
        ns = 2
    else:
        ns = 5
    ns = np.max([2, ns])
    rangeint = np.linspace(0,1,ns)
    for timeinst in range(rangeint.shape[0]-1):
        ti0 = rangeint[timeinst]
        ti1 = rangeint[timeinst+1]
        start = pars[:,0] + pars[:,1]*ti0 + pars[:,2]*(ti0*ti0)
        end = pars[:,0] + pars[:,1]*ti1 + pars[:,2]*(ti1*ti1)
        start = np.round(start).astype(np.int32)
        end = np.round(end).astype(np.int32)
        rr, cc, val = line_aa(start[0], start[1], end[0], end[1])
        valid = np.logical_and(np.logical_and(rr < H.shape[0], cc < H.shape[1]), np.logical_and(rr > 0, cc > 0))
        rr = rr[valid]
        cc = cc[valid]
        val = val[valid]
        if len(H.shape) > 2:
            H[rr, cc, 0] = 0
            H[rr, cc, 1] = 0
            H[rr, cc, 2] = val
        else:
            H[rr, cc] = val 
    return H

def shapeblur_addbg(folder):
    n_im = 5
    background_images = os.listdir(g_background_image_path)
    all_objects = glob.glob(folder+'/*')
    all_objects.sort()
    k = 5
    ki = 9
    for obj in all_objects[ki*k:k*(ki+1)]:
        print("From {} to {}".format(ki*k,k*(ki+1)))
        print(obj)
        diffbgrpath = obj + '/diffbgr/' 
        if not os.path.exists(diffbgrpath):
            os.makedirs(diffbgrpath)
        all_objid = glob.glob(obj+'/GT/*')
        all_objid.sort()
        for objid in all_objid:
            if int(objid[-4:]) % 50 == 1:
                print(objid)
            # pdb.set_trace()

            seq_name = random.choice(background_images)
            seq_images = glob.glob(os.path.join(g_background_image_path,seq_name,"*.jpg"))
            if len(seq_images) <= n_im:
                seq_images = glob.glob(os.path.join(g_background_image_path,seq_name,"*.png"))
            seq_images.sort()
            bgri = random.randint(n_im,len(seq_images)-1)
            bgr_path = seq_images[bgri]

            B0 = cv2.imread(bgr_path)/255
            B = cv2.resize(B0, dsize=(int(g_resolution_x*g_resolution_percentage/100), int(g_resolution_y*g_resolution_percentage/100)), interpolation=cv2.INTER_CUBIC)
            B[B > 1] = 1
            B[B < 0] = 0
            FH = np.zeros(B.shape)
            MH = np.zeros(B.shape[:2])
            pars = np.array([[(B.shape[0]-1)/2-1, (B.shape[1]-1)/2-1], [1.0, 1.0]]).T
            FM = np.zeros(B.shape[:2]+(4,g_fmo_steps,))
            centroids = np.zeros((2,g_fmo_steps))
            for ki in range(g_fmo_steps):
                FM[:,:,:,ki] = cv2.imread(os.path.join(objid,'image-{:06d}.png'.format(ki+1)),cv2.IMREAD_UNCHANGED)/g_rgb_color_max
                props = regionprops((FM[:,:,-1,ki]>0).astype(int))
                if len(props) != 1:
                    print('Error')
                    return False
                centroids[:,ki] = props[0].centroid
            for ki in range(g_fmo_steps):
                F = FM[:,:,:-1,ki]*FM[:,:,-1:,ki]
                M = FM[:,:,-1,ki]
                if ki < g_fmo_steps-1:
                    pars[:,1] = centroids[:,ki+1] - centroids[:,ki]
                H = renderTraj(pars, np.zeros(B.shape[:2]))
                H /= H.sum()*g_fmo_steps
                for kk in range(3): 
                    FH[:,:,kk] += signal.fftconvolve(H, F[:,:,kk], mode='same')
                MH += signal.fftconvolve(H, M, mode='same')
            Im = FH + (1 - MH)[:,:,np.newaxis]*B
            Im[Im > 1] = 1
            Im[Im < 0] = 0
          
            Im = Im[:,:,[2,1,0]]
            Ims = Image.fromarray((Im * 255).astype(np.uint8))

            pathsave = diffbgrpath + objid[-4:]
            Ims.save(pathsave+'_im.png')

            Ball = np.zeros(B.shape+(n_im,))
            Ball[:,:,:,0] = B
            for ki in range(1,n_im):
                bgrki_path = seq_images[bgri-ki]
                Ball[:,:,:,ki] = cv2.resize(cv2.imread(bgrki_path)/255, dsize=(int(g_resolution_x*g_resolution_percentage/100), int(g_resolution_y*g_resolution_percentage/100)), interpolation=cv2.INTER_CUBIC)
            Ball[Ball > 1] = 1
            Ball[Ball < 0] = 0
            Bmed = np.median(Ball,3)
            Image.fromarray((B[:,:,[2,1,0]] * 255).astype(np.uint8)).save(pathsave+'_bgr.png')
            Image.fromarray((Bmed[:,:,[2,1,0]] * 255).astype(np.uint8)).save(pathsave+'_bgrmed.png')
