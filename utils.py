import numpy as np
import math 

# from deblatting_python.gpu.deblatting_gpu import *
# from deblatting_python.deblatting_pw import *
from main_settings import *

from skimage.draw import line_aa
from skimage import measure
import skimage.transform
from scipy import signal
from skimage.measure import label, regionprops
# from skimage.measure import compare_ssim as ssim
import skimage.metrics as metrics
import scipy.misc
import cv2
import pdb

def deblatting_oracle_runner(img,bgrr,debl_dim,gt_traj):
	nsplits = gt_traj.shape[1]
	Hso = np.zeros(img.shape[:2]+(nsplits,))

	for ni in range(nsplits): 
		if ni < nsplits-1:
			pars = np.array([gt_traj[:,ni], gt_traj[:,ni+1]-gt_traj[:,ni]]).T
		else:
			pars = np.array([gt_traj[:,ni], gt_traj[:,ni]-gt_traj[:,ni-1]]).T
		Hso[:,:,ni] = renderTraj(pars, Hso[:,:,ni])
	Hso /= Hso.sum()

	Fs,Ms = estimateFM_pw(img,bgrr,Hso,np.zeros(tuple(debl_dim)+(nsplits,)))
	rgba_tbd3d = np.concatenate((Fs, Ms),2)
	return rgba_tbd3d, Hso

def deblatting_single_runner(imr,bgrr,nsplits,debl_dim):
	dI = imr.transpose(2,0,1)[np.newaxis,:,:,:]
	dB = bgrr.transpose(2,0,1)[np.newaxis,:,:,:]
	M0 = np.zeros(debl_dim)
	H,F,M = estimateFMH_gpu(dI, dB, M0)
	return H,F,M

def deblatting_runner(imr,bgrr,nsplits,debl_dim):
	dI = imr.transpose(2,0,1)[np.newaxis,:,:,:]
	dB = bgrr.transpose(2,0,1)[np.newaxis,:,:,:]
	M0 = np.zeros(debl_dim)
	H,F,M = estimateFMH_gpu(dI, dB, M0)
	Fc = F.cpu().numpy()[0].transpose(1,2,0)
	Mc = M.cpu().numpy()[0].transpose(1,2,0)
	Hc = H.cpu().numpy()[0,0]
	Hc /= Hc.sum()
	Hf,pars = psffit(Hc,True)
	Fc,Mc = estimateFM(imr,bgrr,Hf,Mc[:,:,0])
	mynorm = np.linalg.norm(pars[:,1])
	if mynorm < 1:
		red_nsplits = 1
	else:
		pcd = nsplits
		while pcd % 2 == 0 and pcd > 0: pcd = pcd // 2 
		red_nsplits = pcd*int(np.min([nsplits/pcd, np.max([1,2**int(np.log2(mynorm))])]))
	Hs = psfsplit(Hc,red_nsplits)
	Fs,Ms = estimateFM_pw(imr,bgrr,Hs,np.zeros(tuple(debl_dim)+(red_nsplits,)))
	inds = np.repeat(range(red_nsplits), int(nsplits/red_nsplits))
	Hs = Hs[:,:,inds]
	Fs = Fs[:,:,:,inds]
	Ms = Ms[:,:,:,inds]
	est_hs_tbd = np.zeros(imr.shape+(nsplits,))
	est_hs_tbd3d = np.zeros(imr.shape+(nsplits,))
	est_traj = np.zeros((2,nsplits))
	timestamps = np.linspace(0,1,nsplits)
	for ki in range(nsplits): 
		Hsc = Hs[:,:,ki]/np.sum(Hs[:,:,ki])
		est_hs_tbd[:,:,:,ki] = fmo_model(bgrr,Hsc,Fc,Mc)
		est_hs_tbd3d[:,:,:,ki] = fmo_model(bgrr,Hsc,Fs[:,:,:,ki],Ms[:,:,0,ki])
		est_traj[:,ki] = pars[:,0] + timestamps[ki]*pars[:,1]
	rgba_tbd = np.repeat(np.concatenate((Fc, Mc[:,:,None]),2)[...,None], nsplits, 3)
	rgba_tbd3d = np.concatenate((Fs, Ms),2)
	return est_hs_tbd, est_hs_tbd3d, rgba_tbd, rgba_tbd3d, est_traj[[1,0]], Hs

def calculate_psnr(img1, img2, max_value=1):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    if len(img2.shape) == 3:
    	mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2[:,:,:,None], dtype=np.float32)) ** 2)
    else:
    	mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)

    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))
 
def calculate_ssim(img1, img2):
    """"Calculating Structural similarity index (SSIM) between two images."""
    if len(img1.shape) == 3:
    	return metrics.structural_similarity(img1, img2, data_range=img2.max() - img2.min())
    else:
    	ssims = np.zeros(img1.shape[3]) 
    	for kk in range(ssims.shape[0]):
    		if len(img2.shape) == 4:
    			ssims[kk] = metrics.structural_similarity(img1[:,:,:,kk], img2[:,:,:,kk], data_range=img2.max() - img2.min(),multichannel=True)
    		else:
    			ssims[kk] = metrics.structural_similarity(img1[:,:,:,kk], img2, data_range=img2.max() - img2.min(),multichannel=True)

    	return np.mean(ssims)

def fmo_detect(I,B):
	## simulate FMO detector -> find approximate location of FMO
	dI = (np.sum(np.abs(I-B),2) > 0.05).astype(float)
	labeled = label(dI)
	regions = regionprops(labeled)
	ind = -1
	maxsol = 0
	for ki in range(len(regions)):
		if regions[ki].area > 100 and regions[ki].area < 0.01*np.prod(dI.shape):
			if regions[ki].solidity > maxsol:
				ind = ki
				maxsol = regions[ki].solidity
	if ind == -1:
		return [], 0
	
	# pdb.set_trace()
	bbox = np.array(regions[ind].bbox).astype(int)
	return bbox, regions[ind].minor_axis_length

def imread(name):
	img = cv2.imread(name,cv2.IMREAD_UNCHANGED)
	if img.shape[2] == 3:
		return img[:,:,[2,1,0]]/255
	else:
		return img[:,:,[2,1,0,3]]/65535

def imwrite(im, name = tmp_folder + 'tmp.png'):
	im[im<0]=0
	im[im>1]=1
	cv2.imwrite(name, im[:,:,[2,1,0]]*255)

def fmo_detect_maxarea(I,B,maxarea = 0.1):
	## simulate FMO detector -> find approximate location of FMO
	dI = (np.sum(np.abs(I-B),2) > maxarea).astype(float)
	labeled = label(dI)
	regions = regionprops(labeled)
	ind = -1
	maxarea = 0
	for ki in range(len(regions)):
		if regions[ki].area > maxarea:
			ind = ki
			maxarea = regions[ki].area
	if ind == -1:
		return [], 0
	bbox = np.array(regions[ind].bbox).astype(int)
	return bbox, regions[ind].minor_axis_length

def fmo_detect_hs(gt_hs,B):
	dI = (np.sum((np.sum(np.abs(gt_hs-B[:,:,:,None]),2) > 0.1),2) > 0.5).astype(float)
	labeled = label(dI)
	regions = regionprops(labeled)
	ind = -1
	maxarea = 0
	for ki in range(len(regions)):
		if regions[ki].area > maxarea:
			ind = ki
			maxarea = regions[ki].area
	if ind == -1:
		return [], 0
	bbox = np.array(regions[ind].bbox).astype(int)
	return bbox, regions[ind].minor_axis_length

def bbox_detect_hs(gt_hs,B):
	dI = (np.sum(np.abs(gt_hs-B),2) > 0.1).astype(float)
	labeled = label(dI)
	regions = regionprops(labeled)
	ind = -1
	maxarea = 0
	for ki in range(len(regions)):
		if regions[ki].area > maxarea:
			ind = ki
			maxarea = regions[ki].area
	if ind == -1:
		return []
	bbox = np.array(regions[ind].bbox).astype(int)
	return bbox

def fmo_model(B,H,F,M):
	if len(H.shape) == 2:
		H = H[:,:,np.newaxis]
		F = F[:,:,:,np.newaxis]
	elif len(F.shape) == 3:
		F = np.repeat(F[:,:,:,np.newaxis],H.shape[2],3)
	HM3 = np.zeros(B.shape)
	HF = np.zeros(B.shape)
	for hi in range(H.shape[2]):
		M1 = M
		if len(M.shape) > 2:
			M1 = M[:, :, hi]
		M3 = np.repeat(M1[:, :, np.newaxis], 3, axis=2)
		HM = signal.fftconvolve(H[:,:,hi], M1, mode='same')
		HM3 += np.repeat(HM[:, :, np.newaxis], 3, axis=2)
		F3 = F[:,:,:,hi]
		for kk in range(3):
			HF[:,:,kk] += signal.fftconvolve(H[:,:,hi], F3[:,:,kk], mode='same')
	I = B*(1-HM3) + HF
	return I


def montageF(F):
	return np.reshape(np.transpose(F,(0,1,3,2)),(F.shape[0],-1,F.shape[2]),'F')

def montageH(Hs):
	return np.concatenate((np.sum(Hs[:,:,::3],2)[:,:,np.newaxis], np.sum(Hs[:,:,1::3],2)[:,:,np.newaxis], np.sum(Hs[:,:,2::3],2)[:,:,np.newaxis]),2)

def diskMask(rad):
	sz = 2*np.array([rad, rad])

	ran1 = np.arange(-(sz[1]-1)/2, ((sz[1]-1)/2)+1, 1.0)
	ran2 = np.arange(-(sz[0]-1)/2, ((sz[0]-1)/2)+1, 1.0)
	xv, yv = np.meshgrid(ran1, ran2)
	mask = np.square(xv) + np.square(yv) <= rad*rad
	M = mask.astype(float)
	return M

def boundingBox(img, pads=None):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    if pads is not None:
    	rmin = max(rmin - pads[0], 0)
    	rmax = min(rmax + pads[0], img.shape[0])
    	cmin = max(cmin - pads[1], 0)
    	cmax = min(cmax + pads[1], img.shape[1])
    return rmin, rmax, cmin, cmax
    
def convert_size(size_bytes): 
    if size_bytes == 0: 
        return "0B" 
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB") 
    i = int(math.floor(math.log(size_bytes, 1024)))
    power = math.pow(1024, i) 
    size = round(size_bytes / power, 2) 
    return "{} {}".format(size, size_name[i])

def calc_tiou(gt_traj, traj, rad):
	ns = gt_traj.shape[1]
	est_traj = np.zeros(gt_traj.shape)
	if traj.shape[0] == 4:
		for ni, ti in zip(range(ns), np.linspace(0,1,ns)):
			est_traj[:,ni] = traj[[1,0]]*(1-ti) + ti*traj[[3,2]]
	else:
		bline = (np.abs(traj[3]+traj[7]) > 1.0).astype(float)
		if bline:
			len1 = np.linalg.norm(traj[[5,1]])
			len2 = np.linalg.norm(traj[[7,3]])
			v1 = traj[[5,1]]/len1
			v2 = traj[[7,3]]/len2
			piece = (len1+len2)/(ns-1)
			for ni in range(ns):
				est_traj[:,ni] = traj[[4,0]] + np.min([piece*ni, len1])*v1 + np.max([0,piece*ni-len1])*v2
		else:
			for ni, ti in zip(range(ns), np.linspace(0,1,ns)):
				est_traj[:,ni] = traj[[4,0]] + ti*traj[[5,1]] + ti*ti*traj[[6,2]]
	
	est_traj2 = est_traj[:,-1::-1]

	ious = calciou(gt_traj, est_traj, rad)
	ious2 = calciou(gt_traj, est_traj2, rad)
	return np.max([np.mean(ious), np.mean(ious2)])

def calciou(p1, p2, rad):
	dists = np.sqrt( np.sum( np.square(p1 - p2),0) )
	dists[dists > 2*rad] = 2*rad

	theta = 2*np.arccos( dists/ (2*rad) )
	A = ((rad*rad)/2) * (theta - np.sin(theta))
	I = 2*A
	U = 2* np.pi * rad*rad - I
	iou = I / U
	return iou

def write_trajectory(Img, traj):
	for kk in range(traj.shape[1]-1):
		Img = renderTraj(np.c_[traj[:,kk], traj[:,kk+1]-traj[:,kk]][::-1], Img)
	# Img[traj[1].astype(int),traj[0].astype(int),1] = 1.0
	return Img
	

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


def generate_lowFPSvideo(V,k=8,gamma_coef = 0.4,do_WB=True):
	newk = int(np.floor(V.shape[3]/k))
	Vk = np.reshape(V[:,:,:,:newk*k], (V.shape[0], V.shape[1], V.shape[2], newk, k) ).mean(-1)
	if do_WB:
		WB = np.expand_dims(np.array([2,1,2]),[0,1,3])
		Vk_WB = ((Vk * WB)/WB.max())**gamma_coef
		WB = np.expand_dims(np.array([2,1,2]),[0,1,3])
	else:
		Vk_WB = Vk**gamma_coef
	return Vk_WB

def extend_bbox(bbox,ext,aspect_ratio,shp):
	height, width = bbox[2] - bbox[0], bbox[3] - bbox[1]
			
	h2 = height + ext

	h2 = int(np.ceil(np.ceil(h2 / aspect_ratio) * aspect_ratio))
	w2 = int(h2 / aspect_ratio)

	wdiff = w2 - width
	wdiff2 = int(np.round(wdiff/2))
	hdiff = h2 - height
	hdiff2 = int(np.round(hdiff/2))

	bbox[0] -= hdiff2
	bbox[2] += hdiff-hdiff2
	bbox[1] -= wdiff2
	bbox[3] += wdiff-wdiff2
	bbox[bbox < 0] = 0
	bbox[2] = np.min([bbox[2], shp[0]-1])
	bbox[3] = np.min([bbox[3], shp[1]-1])
	return bbox

def extend_bbox_uniform(bbox,ext,shp):
	bbox[0] -= ext
	bbox[2] += ext
	bbox[1] -= ext
	bbox[3] += ext
	bbox[bbox < 0] = 0
	bbox[2] = np.min([bbox[2], shp[0]-1])
	bbox[3] = np.min([bbox[3], shp[1]-1])
	return bbox

def extend_bbox_nonuniform(bbox,ext,shp):
	bbox[0] -= ext[0]
	bbox[2] += ext[0]
	bbox[1] -= ext[1]
	bbox[3] += ext[1]
	bbox[bbox < 0] = 0
	bbox[2] = np.min([bbox[2], shp[0]-1])
	bbox[3] = np.min([bbox[3], shp[1]-1])
	return bbox

def bbox_fmo(bbox,gt_hs,B):
	gt_hs_crop = crop_only(gt_hs,bbox)
	B_crop = crop_only(B,bbox)
	bbox_crop,rad = fmo_detect_hs(gt_hs_crop,B_crop)
	bbox_new = bbox_crop
	if len(bbox_new) > 0:
		bbox_new[:2] += bbox[:2]
		bbox_new[2:] += bbox[:2]
	else:
		bbox_new = bbox
	return bbox_new

def rgba2hs(rgba, bgr):
	return rgba[:,:,:3]*rgba[:,:,3:] + bgr[:,:,:,None]*(1-rgba[:,:,3:])

def rgba2rgb(rgba):
	return rgba[:,:,:3]*rgba[:,:,3:] + 1*(1-rgba[:,:,3:])

def sync_directions(est_hs_crop, gt_hs_crop):
	do_flip = False
	if gt_hs_crop is not None:
		if np.mean((est_hs_crop[:,:,:,0] - gt_hs_crop[:,:,:,0])**2) > np.mean((est_hs_crop[:,:,:,0] - gt_hs_crop[:,:,:,-1])**2):
			est_hs_crop = est_hs_crop[:,:,:,::-1]
			do_flip = True
	return est_hs_crop, do_flip

def	sync_directions_smooth(est_hs_crop, est_traj, est_traj_prev, radius):
	if est_traj_prev is not None:
		dist = np.min([np.linalg.norm(est_traj[:,0] - est_traj_prev[:,0]), np.linalg.norm(est_traj[:,0] - est_traj_prev[:,-1])])
		dist2 = np.min([np.linalg.norm(est_traj[:,-1] - est_traj_prev[:,0]), np.linalg.norm(est_traj[:,-1] - est_traj_prev[:,-1])])
		flip_due_to_newobj = np.min([dist,dist2]) > 2*radius and est_traj[1,-1] > est_traj[1,0]
		flip_due_to_smoothness = dist2 < dist
		do_flip = flip_due_to_newobj or flip_due_to_smoothness
	else:
		do_flip = est_traj[1,-1] > est_traj[1,0]
	if do_flip:
		est_hs_crop = est_hs_crop[:,:,:,::-1]
		est_traj = est_traj[:,::-1]
	return est_hs_crop, est_traj, do_flip

def crop_resize(Is, bbox, res):
	if Is is None:
		return None
	rev_axis = False
	if len(Is.shape) == 3:
		rev_axis = True
		Is = Is[:,:,:,np.newaxis]
	imr = np.zeros((res[1], res[0], 3, Is.shape[3]))
	for kk in range(Is.shape[3]):
		im = Is[bbox[0]:bbox[2], bbox[1]:bbox[3], :, kk]
		imr[:,:,:,kk] = cv2.resize(im, res, interpolation = cv2.INTER_CUBIC)
	if rev_axis:
		imr = imr[:,:,:,0]
	return imr

def crop_only(Is, bbox):
	if Is is None:
		return None
	return Is[bbox[0]:bbox[2], bbox[1]:bbox[3]]

def rev_crop_resize_traj(inp, bbox, res):
	inp[0] *= ( (bbox[2]-bbox[0])/res[1])
	inp[1] *= ( (bbox[3]-bbox[1])/res[0])
	inp[0] += bbox[0]
	inp[1] += bbox[1]
	return np.array(inp[[1,0]])

def rev_crop_resize(inp, bbox, I):
	est_hs = np.tile(I.copy()[:,:,:,np.newaxis],(1,1,1,inp.shape[3]))
	for hsk in range(inp.shape[3]):
		est_hs[bbox[0]:bbox[2], bbox[1]:bbox[3],:,hsk] = cv2.resize(inp[:,:,:,hsk], (bbox[3]-bbox[1],bbox[2]-bbox[0]), interpolation = cv2.INTER_CUBIC)
	return est_hs
