import numpy as np
import os, glob
import scipy.io
import cv2
from helpers.torch_helpers import *
from utils import *
import pdb

class GroundTruthProcessor:
	def __init__(self, ff, kkf, folder, medn, shift, update_bg = True):
		self.nsplits = 8
		self.do_wb = False
		self.update_bg = update_bg
		self.shift = shift
		roi_frames = []
		Vk = []
		if os.path.exists(folder + 'roi_frames.txt'):
			roi_frames = np.loadtxt(folder + 'roi_frames.txt').astype(int)
		self.syn_mode = False
		if ff == os.path.split(ff)[-1]:
			seqname = ff
			seqpath = folder + seqname + '/'
			nfrms = len(glob.glob(seqpath + '*.png'))
			start_ind = 1
			end_ind = nfrms
			w_trajgt = True
			self.hspath_base = folder + seqname + '/GT/' 
			self.use_hs = True
			self.syn_mode = True
			self.do_wb = False
			Vk = imread(seqpath + "{}_{:04d}.png".format(seqname,start_ind))
		else:
			seqname = os.path.split(ff)[-1][:-4]
			if '-12' in seqname:
				self.nsplits = 12
			seqpath = folder + 'imgs/' + seqname + '/'
			if not os.path.exists(seqpath):
				seqpath = folder + seqname + '/'
				if not os.path.exists(seqpath):
					print('Directory does not exist')
			nfrms = len(glob.glob(seqpath + '*.png'))
			start_ind = 0
			end_ind = nfrms
			if roi_frames != []:
				start_ind = roi_frames[kkf,0]
				end_ind = roi_frames[kkf,1]
				nfrms = end_ind - start_ind + 1
			if not os.path.exists(seqpath + "{:08d}.png".format(0)):
				start_ind += 1
			mednused = np.min([medn, nfrms])
			for kk in range(mednused):
				path = seqpath + "{:08d}.png".format(start_ind+kk)
				Im = imread(path)
				Im = Im[shift:]
				if Vk == []:
					Vk = np.zeros(Im.shape+(mednused,))
				Vk[:,:,:,kk] = Im
			if self.do_wb:
				Vk = generate_lowFPSvideo(Vk,k=1)
			pars = []
			w_trajgt = False
			if os.path.exists(seqpath + 'gt.txt'):
				w_trajgt = True
				pars = np.loadtxt(seqpath + 'gt.txt')
			rads = []
			if os.path.exists(seqpath + 'gtr.txt'):
				rads = np.loadtxt(seqpath + 'gtr.txt')
			elif os.path.exists(folder + 'templates/'):
				template_path = folder + 'templates/'+ seqname.replace('-gc','') + '_template.mat'
				data = scipy.io.loadmat(template_path)
				template = data['template']
				rads = (template.shape[0]/2)/data['scale']
			if not w_trajgt and os.path.exists(folder + 'gt_bbox/' + seqname + '.txt'):
				w_trajgt = True
				bboxes = np.loadtxt(folder + 'gt_bbox/' + seqname + '.txt')
				pars = np.reshape(bboxes[:,:2] + 0.5*bboxes[:,2:], (-1,self.nsplits,2)).transpose((0,2,1))
				pars[:,1,:] -= shift
				pars = np.reshape(pars,(-1,self.nsplits))
				rads = np.reshape(np.max(0.5*bboxes[:,2:],1), (-1,self.nsplits))
				pars = np.r_[np.zeros((start_ind*2,self.nsplits)),pars]
				rads = np.r_[np.zeros((start_ind,self.nsplits)),rads]
			self.hspath_base = folder + 'imgs_gt/' + seqname + '/'
			if not os.path.exists(self.hspath_base + "{:08d}.png".format(1)):
				self.hspath_base = folder + 'imgs_gt/' + seqname.replace('_newGT','')[3:-5] + '/'
			if not os.path.exists(self.hspath_base + "{:08d}.png".format(1)):
				self.hspath_base = folder + 'imgs_gt/' + seqname.replace('_newGT','')[3:-6] + '/'
			self.use_hs = os.path.exists(self.hspath_base + "{:08d}.png".format(1))
			self.start_zero = 1-int(os.path.exists(self.hspath_base + "{:08d}.png".format(0)))
			self.pars = pars
			self.rads = rads
			self.mednused = mednused
		self.start_ind = start_ind
		self.nfrms = nfrms
		self.Vk = Vk
		self.seqname = seqname
		self.seqpath = seqpath
		self.w_trajgt = w_trajgt
		print('Sequence {} has {} frames'.format(seqname, nfrms))

			# get_training_sample(["can"],min_obj=5,max_obj=5)

	def get_img_noupd(self, kk):
		if self.syn_mode:
			path = self.seqpath + "{}_{:04d}.png".format(self.seqname,self.start_ind+kk)
		else:
			path = self.seqpath + "{:08d}.png".format(self.start_ind+kk)
		I = imread(path)
		I = I[self.shift:]
		return I

	def get_img(self, kk):
		if self.syn_mode:
			path = self.seqpath + "{}_{:04d}.png".format(self.seqname,self.start_ind+kk)
		else:
			path = self.seqpath + "{:08d}.png".format(self.start_ind+kk)
		I = imread(path)
		if self.do_wb:
			I = generate_lowFPSvideo(I[...,None],k=1)[...,0]
		I = I[self.shift:]
		if self.syn_mode:
			path = self.seqpath + "GT/{}_{:04d}/bgr_med.png".format(self.seqname,self.start_ind+kk)
			B = imread(path)
		else:
			B = np.median(self.Vk, 3)
			if kk >= self.mednused and self.update_bg:
				self.Vk[:,:,:,:-1] = self.Vk[:,:,:,1:]
				self.Vk[:,:,:,-1] = I
		return I,B

	def get_hs(self, kk):
		if self.use_hs:
			chnls = 3
			if self.syn_mode:
				chnls = 4
			Vos = np.zeros((self.Vk.shape[0], self.Vk.shape[1], chnls, self.nsplits))
			for hsk in range(self.nsplits):
				if self.syn_mode:
					hspath = self.hspath_base + "{}_{:04d}/image-{:06d}.png".format(self.seqname,self.start_ind+kk,3*(self.start_ind+hsk))
				else:
					hspath = self.hspath_base + "{:08d}.png".format((kk+self.start_ind)*self.nsplits + hsk + self.start_zero)
				Vo = imread(hspath)
				Vo = Vo[self.shift:]
				Vos[:,:,:,hsk] = Vo[:self.Vk.shape[0],:self.Vk.shape[1]]
			if self.do_wb:
				Vos = generate_lowFPSvideo(Vos,k=1)
			return Vos
		return None

	def get_trajgt(self, kk):
		if self.syn_mode:
			Vos = self.get_hs(kk)
			B0 = np.zeros(Vos.shape[:2]+(3,))
			pars = np.zeros((2,Vos.shape[3]))
			for ki in range(pars.shape[1]):
				bbox0 = bbox_detect_hs(Vos[:,:,[3,3,3],ki],B0)
				pars[:,ki] = (bbox0[:2] + bbox0[2:])/2
			bbox, radius = fmo_detect_hs(Vos[:,:,[3,3,3],:],B0)
			bbox = np.array(bbox).astype(int)
			radius = np.round(radius*1.5).astype(int)
			return pars[[1,0]], radius, bbox
		elif self.w_trajgt:
			par = self.pars[2*(kk+self.start_ind):2*(kk+self.start_ind+1),:].T
			self.nsplits = par.shape[0]
			parsum = par.sum(1)
			nans = np.isnan(parsum)
			if nans.sum() > 0:
				ind = np.nonzero(nans)[0]
				for indt in ind:
					if indt == 0:
						par[indt,:] = par[np.nonzero(~nans)[0][0],:]
					elif indt < self.nsplits-1 and not nans[indt+1]:
						par[indt,:] = (par[indt-1,:] + par[indt+1,:])/2
					else:
						par[indt,:] = par[indt-1,:]

			bbox = (par[:,1].min(), par[:,0].min(), par[:,1].max(), par[:,0].max())
			if self.rads.shape[0] > 1:
				radius = np.round(np.nanmax(self.rads[self.start_ind+kk,:])).astype(int)
			else:
				radius = np.round(self.rads[0,0]).astype(int)
			bbox = np.array(bbox).astype(int)
			return par.T, radius, bbox
		return None, None, None

#######################################################################################################################
#######################################################################################################################

class SequenceLogger:
	def __init__(self, log_folder, gtp, algname=''):
		self.writepath = log_folder + gtp.seqname + '/'
		if not os.path.exists(self.writepath):
			os.makedirs(self.writepath)
		self.ImGT = gtp.Vk[:,:,:,0].copy()
		self.ImEst = gtp.Vk[:,:,:,0].copy()
		self.nsplits = gtp.nsplits
		self.save_superres = True
		self.algname = algname
		if self.save_superres:
			self.srwriter = SRWriter(self.ImGT, self.writepath+self.algname+'sr.avi',gtp.use_hs)

	def write_trajgt(self, gt_traj):
		write_trajectory(self.ImGT, gt_traj)
		# save_image(self.preprocess(self.ImGT).clone(),self.writepath + 'imgt.png')
		imwrite(self.ImGT,self.writepath + 'imgt.png')

	def write_trajest(self, est_traj):
		write_trajectory(self.ImEst, est_traj)
		# save_image(self.preprocess(self.ImEst).clone(),self.writepath + 'imest.png')
		imwrite(self.ImEst,self.writepath+self.algname + 'imest.png')

	def write_crops(self,kk,renders_rgb, est_hs_crop, gt_hs_crop, im_crop, bgr_crop):
		videoname = '{:04d}video{}.avi'.format(kk,self.algname)
		shp = (im_crop.shape[0]*2, im_crop.shape[1]*2, 3)
		imw = np.zeros(shp)
		imw[im_crop.shape[0]:,im_crop.shape[1]:] = im_crop
		out = cv2.VideoWriter(os.path.join(self.writepath, videoname),cv2.VideoWriter_fourcc(*"MJPG"), 6, (shp[1], shp[0]), True)
		for ki in range(est_hs_crop.shape[3]):
			imw[:im_crop.shape[0],:im_crop.shape[1]] = est_hs_crop[:,:,:,ki]
			imw[im_crop.shape[0]:,:im_crop.shape[1]] = renders_rgb[:,:,:,ki]
			if gt_hs_crop is not None:
				imw[:im_crop.shape[0],im_crop.shape[1]:] = gt_hs_crop[:,:,:,ki]
			imw[imw>1]=1
			imw[imw<0]=0
			out.write( (imw[:,:,[2,1,0]] * 255).astype(np.uint8) )
		out.release()

	def write_crops_3c(self,kk,renders_rgb, est_hs_crop, gt_hs_crop, im_crop, bgr_crop):
		videoname = '{:04d}video{}_3c.avi'.format(kk,self.algname)
		if gt_hs_crop is None:
			fctr = 2
		else:
			fctr = 3
		shp = (im_crop.shape[0], im_crop.shape[1]*fctr, 3)
		imw = np.zeros(shp)
		imw[:,:im_crop.shape[1]] = im_crop
		out = cv2.VideoWriter(os.path.join(self.writepath, videoname),cv2.VideoWriter_fourcc(*"MJPG"), 6, (shp[1], shp[0]), True)
		for ki in range(est_hs_crop.shape[3]):
			imw[:,im_crop.shape[1]:2*im_crop.shape[1]] = est_hs_crop[:,:,:,ki]
			if not gt_hs_crop is None:
				imw[:,2*im_crop.shape[1]:3*im_crop.shape[1]] = gt_hs_crop[:,:,:,ki]
			imw[imw>1]=1
			imw[imw<0]=0
			out.write( (imw[:,:,[2,1,0]] * 255).astype(np.uint8) )
		out.release()

	def write_crops_4c(self,kk,renders_rgb, est_hs_crop, gt_hs_crop, im_crop, bgr_crop, est_gt_hs):
		videoname = '{:04d}video{}_4c.avi'.format(kk,self.algname)
		fctr = 4
		shp = (im_crop.shape[0], im_crop.shape[1]*fctr, 3)
		imw = np.zeros(shp)
		imw[:,:im_crop.shape[1]] = im_crop
		out = cv2.VideoWriter(os.path.join(self.writepath, videoname),cv2.VideoWriter_fourcc(*"MJPG"), 6, (shp[1], shp[0]), True)
		for ki in range(est_hs_crop.shape[3]):
			imw[:,im_crop.shape[1]:2*im_crop.shape[1]] = est_hs_crop[:,:,:,ki]
			imw[:,2*im_crop.shape[1]:3*im_crop.shape[1]] = gt_hs_crop[:,:,:,ki]
			imw[:,3*im_crop.shape[1]:4*im_crop.shape[1]] = est_gt_hs[:,:,:,ki]
			imw[imw>1]=1
			imw[imw<0]=0
			out.write( (imw[:,:,[2,1,0]] * 255).astype(np.uint8) )
		out.release()


	def write_superres(self, I, est_hs, Vos):
		if self.save_superres:
			self.srwriter.update_ls(I)
			for hsk in range(self.nsplits):
				vosframe = None
				if Vos is not None:
					vosframe = Vos[:,:,:,hsk]
				if est_hs is None:
					self.srwriter.write_next(vosframe,I) 
				else:
					self.srwriter.write_next(vosframe,est_hs[:,:,:,hsk]) 

	def close(self):
		if self.save_superres:
			self.srwriter.close()

#######################################################################################################################
#######################################################################################################################

class AverageScoreTracker:
	def __init__(self, nfiles, algname='DeFMO', verbose=True):
		self.av_ious = np.zeros(nfiles)
		self.av_psnr = np.zeros(nfiles)
		self.av_ssim = np.zeros(nfiles)
		self.av_ipsnr = np.zeros(nfiles)
		self.av_issim = np.zeros(nfiles)
		self.av_bpsnr = np.zeros(nfiles)
		self.av_bssim = np.zeros(nfiles)
		self.av_times = []
		self.seqi = 0
		self.algname = algname
		self.verbose = verbose

	def next(self, seqname, sst):
		self.av_ious[self.seqi] = np.mean(sst.all_ious[sst.all_ious > 0])
		self.av_psnr[self.seqi] = np.mean(sst.all_psnr[sst.all_psnr > 0])
		self.av_ssim[self.seqi] = np.mean(sst.all_ssim[sst.all_ssim > 0])
		self.av_ipsnr[self.seqi] = np.mean(sst.imr_psnr[sst.imr_psnr > 0])
		self.av_issim[self.seqi] = np.mean(sst.imr_ssim[sst.imr_ssim > 0])
		self.av_bpsnr[self.seqi] = np.mean(sst.bgr_psnr[sst.bgr_psnr > 0])
		self.av_bssim[self.seqi] = np.mean(sst.bgr_ssim[sst.bgr_ssim > 0])
		print('{}: Finished seq {}, average tiou {:.3f}, psnr {:.3f} dB, ssim {:.3f}'.format(self.algname,seqname, self.av_ious[self.seqi], self.av_psnr[self.seqi], self.av_ssim[self.seqi]))
		print('Imr: psnr {:.3f} dB, ssim {:.3f}'.format(self.av_ipsnr[self.seqi], self.av_issim[self.seqi]))
		print('Bgr: psnr {:.3f} dB, ssim {:.3f}'.format(self.av_bpsnr[self.seqi], self.av_bssim[self.seqi]))
		self.seqi += 1

	def next_time(self, tm):
		self.av_times.append(tm)

	def close(self):
		print('AVERAGES')
		print('{}: tiou {:.3f}, psnr {:.3f} dB, ssim {:.3f}'.format(self.algname, np.nanmean(self.av_ious), np.nanmean(self.av_psnr), np.nanmean(self.av_ssim)))
		print('{}: time {:.3f} seconds'.format(self.algname, np.nanmean(np.array(self.av_times))))
		if self.verbose:
			print('Imr: psnr {:.3f} dB, ssim {:.3f}'.format(np.nanmean(self.av_ipsnr), np.nanmean(self.av_issim)))
			print('Bgr: psnr {:.3f} dB, ssim {:.3f}'.format(np.nanmean(self.av_bpsnr), np.nanmean(self.av_bssim)))

#######################################################################################################################
#######################################################################################################################

class SequenceScoreTracker:
	def __init__(self, nfrms, algname='DeFMO',verbose=True):
		self.all_ious = np.zeros(nfrms)
		self.all_psnr = np.zeros(nfrms)
		self.all_ssim = np.zeros(nfrms)
		self.imr_psnr = np.zeros(nfrms)
		self.imr_ssim = np.zeros(nfrms)
		self.bgr_psnr = np.zeros(nfrms)
		self.bgr_ssim = np.zeros(nfrms)
		self.algname = algname
		self.verbose = verbose

	def next_traj(self,kk,gt_traj,est_traj,minor_axis_length):
		ious = calciou(gt_traj, est_traj, minor_axis_length)
		ious2 = calciou(gt_traj, est_traj[:,-1::-1], minor_axis_length)
		iou = np.max([np.mean(ious), np.mean(ious2)])
		self.all_ious[kk] = iou
		return iou

	def next_appearance(self,kk,gt_hs,est_hs,im,bgr):
		self.all_psnr[kk] = calculate_psnr(gt_hs, est_hs)
		self.all_ssim[kk] = calculate_ssim(gt_hs, est_hs)
		self.imr_psnr[kk] = calculate_psnr(gt_hs, im)
		self.imr_ssim[kk] = calculate_ssim(gt_hs, im)
		self.bgr_psnr[kk] = calculate_psnr(gt_hs, bgr)
		self.bgr_ssim[kk] = calculate_ssim(gt_hs, bgr)

	def report(self, seqname, kk):
		if self.verbose:
			print('{}: Seq {}, frm {}, iou {:.3f}, psnr {:.3f}, ssim {:.3f}'.format(self.algname, seqname, kk, self.all_ious[kk], self.all_psnr[kk], self.all_ssim[kk]))
			print('Imr: psnr {:.3f}, ssim {:.3f}'.format(self.imr_psnr[kk], self.imr_ssim[kk]))
			print('Bgr: psnr {:.3f}, ssim {:.3f}'.format(self.bgr_psnr[kk], self.bgr_ssim[kk]))
		else:
			print('{}: iou {:.3f}, psnr {:.3f}, ssim {:.3f}'.format(self.algname, self.all_ious[kk], self.all_psnr[kk], self.all_ssim[kk]))
