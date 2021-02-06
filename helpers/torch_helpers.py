import torch
from torch.nn import functional as F
from torchvision.utils import save_image
from skimage.measure import label, regionprops
import os
import cv2
import numpy as np
from main_settings import *
import matplotlib.pyplot as plt
from PIL import Image
import pdb

class SRWriter:
	def __init__(self, imtemp, path, available_gt=True):
		self.available_gt = available_gt
		if self.available_gt:
			fctr = 3
		else:
			fctr = 2
		if imtemp.shape[0] > imtemp.shape[1]:
			self.width = True
			shp = (imtemp.shape[0], imtemp.shape[1]*fctr, 3)
			self.value = imtemp.shape[1]
		else:
			self.width = False
			shp = (imtemp.shape[0]*fctr, imtemp.shape[1], 3)
			self.value = imtemp.shape[0]
		self.video = cv2.VideoWriter(path,cv2.VideoWriter_fourcc(*"MJPG"), 12, (shp[1], shp[0]), True)
		self.img = np.zeros(shp)

	def update_ls(self,lsf):
		if self.width:
			self.img[:,:self.value] = lsf
		else:
			self.img[:self.value,:] = lsf

	def write_next(self,hs,est):
		if hs is not None:
			if self.width:
				self.img[:,2*self.value:] = hs
			else:
				self.img[2*self.value:,:] = hs
		if est is not None:
			if self.width:
				self.img[:,self.value:2*self.value] = est
			else:
				self.img[self.value:2*self.value,:] = est
		self.img[self.img>1]=1
		self.img[self.img<0]=0
		self.video.write( (self.img.copy() * 255)[:,:,[2,1,0]].astype(np.uint8) )

	def close(self):
		self.video.release()

def renders2traj(renders,device):
	masks = renders[:,:,-1]
	sumx = torch.sum(masks,-2)
	sumy = torch.sum(masks,-1)
	cenx = torch.sum(sumy*torch.arange(1,sumy.shape[-1]+1)[None,None].float().to(device),-1) / torch.sum(sumy,-1)
	ceny = torch.sum(sumx*torch.arange(1,sumx.shape[-1]+1)[None,None].float().to(device),-1) / torch.sum(sumx,-1)
	est_traj = torch.cat((cenx.unsqueeze(-1),ceny.unsqueeze(-1)),-1)
	return est_traj

def renders2traj_bbox(renders_rgba):
	masks = renders_rgba[:,:,-1]
	est_traj = []
	for ti in range(masks.shape[2]):
		th = np.min([0.1, 0.5*np.max(masks[:,:,ti])])
		dI = (masks[:,:,ti] >= th).astype(float)
		labeled = label(dI)
		regions = regionprops(labeled)
		areas = [reg.area for reg in regions]
		region = regions[np.argmax(areas)]
		bbox = np.array(region.bbox)
		est_traj = np.r_[est_traj, bbox[:2] + (bbox[2:]-bbox[:2])/2]
	est_traj = np.reshape(est_traj, (-1,2)).T
	return est_traj
	
def write_latent(rendering, latent, device, folder=g_temp_folder,steps=g_fmo_steps,videoname='output.avi'):
	write_video = True
	write_images = False
	eps = 0
	out = None
	with torch.no_grad():
		times = torch.linspace(0+eps,1-eps,steps).to(device)
		renders = rendering(latent,times[None])
		for ki in range(renders.shape[1]):
			ti = times[ki]
			if write_images:
				save_image(renders[0,ki].clone(), os.path.join(folder, 'latent{:04d}.png'.format(int(ti*100))))
			if write_video:
				if out is None:
					out = cv2.VideoWriter(os.path.join(folder, videoname),cv2.VideoWriter_fourcc(*"MJPG"), 6, (renders.shape[4], renders.shape[3]),True)
				im4 = renders[0,ki].data.cpu().detach().numpy().transpose(1,2,0)
				im = im4[:,:,[2,1,0]] * im4[:,:,3:] + 1* (1 - im4[:,:,3:])
				out.write( (im * 255).astype(np.uint8) )
	if write_video:
		out.release()

	return renders

def write_gt(gt_paths, folder=g_temp_folder, bgr_clr = 1):
	write_video = True
	out = None
	renders = []
	for ti in range(len(gt_paths)):
		im4 = np.array(Image.open(gt_paths[ti]))/255
		renders.append(im4[np.newaxis].copy())
		if out is None:
			out = cv2.VideoWriter(os.path.join(folder, 'output_gt.avi'),cv2.VideoWriter_fourcc(*"MJPG"), 6, (im4.shape[1], im4.shape[0]),True)
		im = im4[:,:,[2,1,0]] * im4[:,:,3:] + bgr_clr* (1 - im4[:,:,3:])
		out.write( (im.copy() * 255).astype(np.uint8) )
	out.release()
	renders = np.stack(renders,1)
	renders = torch.from_numpy(renders).float().permute(0,1,4,2,3)
	return renders	 

def write_gt_masks(gt_paths, folder=g_temp_folder, bgr_clr = 1):
	write_video = True
	out = None
	renders = []
	for ti in range(len(gt_paths)):
		im4 = np.array(Image.open(gt_paths[ti]))/255
		renders.append(im4[np.newaxis].copy())
		if out is None:
			out = cv2.VideoWriter(os.path.join(folder, 'output_masks_gt.avi'),cv2.VideoWriter_fourcc(*"MJPG"), 6, (im4.shape[1], im4.shape[0]),True)
		im = (im4[:,:,[3,3,3]])
		if bgr_clr == 1:
			im = 1 - im
		out.write( (im.copy() * 255).astype(np.uint8) )
	out.release()
	renders = np.stack(renders,1)
	renders = torch.from_numpy(renders).float().permute(0,1,4,2,3)
	return renders	

def get_figure(encoder, rendering, device, val_batch):
	latent = encoder(val_batch)
	times = [0, 1]

	fig = plt.figure() # figsize=(12, 48)
	nidx = len(times)
	for idx in np.arange(nidx):
		t_tensor = torch.FloatTensor([times[idx]]).to(device).repeat(latent.shape[0], 1, latent.shape[2], latent.shape[3])
		result = rendering(torch.cat((t_tensor,latent),1)).cpu().numpy()

		ax = fig.add_subplot(1, nidx, idx+1, xticks=[], yticks=[])
		plt.imshow(np.transpose(result[0], (1, 2, 0)))
		ax.set_title("t = {}".format(times[idx]))
	return fig

def get_images(encoder, rendering, device, val_batch):
	with torch.no_grad():
		latent = encoder(val_batch)
		times = torch.linspace(0,1,2).to(device)
		renders = rendering(latent,times[None])

	renders = renders.cpu().numpy()
	renders = renders[:,:,3:4]*(renders[:,:,:3]-1)+1
	return renders

def normalized_cross_correlation_channels(image1, image2):
	mean1 = image1.mean([2,3,4],keepdims=True)
	mean2 = image2.mean([2,3,4],keepdims=True) 
	std1 = image1.std([2,3,4],unbiased=False,keepdims=True)
	std2 = image2.std([2,3,4],unbiased=False,keepdims=True)
	eps=1e-8
	bs, ts, *sh = image1.shape
	N = sh[0]*sh[1]*sh[2]
	im1b = ((image1-mean1)/(std1*N+eps)).view(bs*ts, sh[0], sh[1], sh[2])
	im2b = ((image2-mean2)/(std2+eps)).reshape(bs*ts, sh[0], sh[1], sh[2])
	padding = tuple(side // 10 for side in sh[:2]) + (0,)
	result = F.conv3d(im1b[None], im2b[:,None], padding=padding, bias=None, groups=bs*ts)
	ncc = result.view(bs*ts, -1).max(1)[0].view(bs, ts)
	return ncc

def normalized_cross_correlation(image1, image2):
	mean1 = image1.mean([2,3],keepdims=True)
	mean2 = image2.mean([2,3],keepdims=True) 
	std1 = image1.std([2,3],unbiased=False,keepdims=True)
	std2 = image2.std([2,3],unbiased=False,keepdims=True)
	eps=1e-8
	bs, ts, *sh = image1.shape
	N = sh[0]*sh[1]
	im1b = ((image1-mean1)/(std1*N+eps)).view(bs*ts, sh[0], sh[1])
	im2b = ((image2-mean2)/(std2+eps)).reshape(bs*ts, sh[0], sh[1])
	padding = tuple(side // 10 for side in sh)
	result = F.conv2d(im1b[None], im2b[:,None], padding=padding, bias=None, groups=bs*ts)
	ncc = result.view(bs*ts, -1).max(1)[0].view(bs, ts)
	return ncc