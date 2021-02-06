import h5py
import os, glob
import numpy as np
import torch

from dataloaders.loader import get_transform
from dataloaders.reporters import *
from main_settings import *
from utils import *
from helpers.torch_helpers import *
import scipy.io
from torchvision.utils import save_image
from PIL import Image, ImageDraw
import cv2
import time

import pdb
from vis import *

def get_tbd_dataset():
	files = np.array(glob.glob(os.path.join(g_tbd_folder, '*-8.mat')))
	files = files[['golf' not in f  for f in files]]
	files = np.r_[files, np.array(glob.glob(os.path.join(g_tbd_folder, '*-12.mat')))]
	files = np.r_[files, np.array(glob.glob(os.path.join(g_tbd_folder, '*-8_newGT.mat')))]
	files = files[['ping_wall' not in f  for f in files]]
	files = files[['fall_coin' not in f  for f in files]]
	files.sort()
	return files

def get_tbd3d_dataset():
	files = np.array(glob.glob(os.path.join(g_tbd3d_folder, '*.mat')))
	if files.shape[0] == 0:
		files = np.array(glob.glob(os.path.join(g_tbd3d_folder, 'imgs/*_GT_*')))
		files = np.array([ff+'.mat' for ff in files])
	files.sort()
	return files

def get_falling_dataset():
	files = np.array(glob.glob(os.path.join(g_falling_folder, '*_GTgamma.mat')))
	if files.shape[0] == 0:
		files = np.array(glob.glob(os.path.join(g_falling_folder, 'imgs/*_GTgamma')))
		files = np.array([ff+'.mat' for ff in files])
	files.sort()
	return files

def get_wildfmo_dataset():
	files = np.array(glob.glob(os.path.join(g_wildfmo_folder, '*.mov')))
	files.sort()
	return files

def get_youtube_dataset():
	files = np.array(glob.glob(os.path.join(g_youtube_folder, '*')))
	files.sort()
	inliers = [not '.' in file for file in files]
	files = files[inliers]
	files = np.array([file+'.avi' for file in files])
	files = files[1:]
	return files

def evaluate_on(encoder, rendering, device, mode = 'tbd'):
	log_folder = tmp_folder+mode+'_eval/'
	medn = 7
	update_bg = True
	verbose = False
	do_defmo = True
	do_deblatting = False
	do_sota18 = False
	do_deblurgan = False
	do_tbdo = False

	eval_d = g_eval_d
	eval_d = 1
	multi_f = 10

	eval_gt = False
	eval_gt_steps = 10

	if do_sota18:
		from helpers.sota18_runner import run_sota18
	if do_deblurgan:
		from helpers.deblurgan_runner import run_deblurgan

	shift = 0
	ext_factor = 4
	if mode == 'tbd':
		files = get_tbd_dataset()
		folder = g_tbd_folder
	elif mode == 'tbd3d':
		files = get_tbd3d_dataset()
		folder = g_tbd3d_folder
		medn = 50
	elif mode == 'tbdfalling':
		files = get_falling_dataset()
		folder = g_falling_folder
		shift = 250
		medn = 50
		ext_factor = 4
	elif mode == 'wildfmo':
		files = get_wildfmo_dataset()
		folder = g_wildfmo_folder
		medn = 50
		ext_factor = 1
	elif mode == 'youtube':
		files = get_youtube_dataset()
		folder = g_youtube_folder
	else:
		print('Mode not found!')

	if do_defmo:
		av_score_tracker = AverageScoreTracker(files.shape)
	if do_deblatting:
		av_score_tracker_tbd = AverageScoreTracker(files.shape,'tbd', False)
		av_score_tracker_tbd3d = AverageScoreTracker(files.shape, 'tbd3d', False)
	if do_tbdo:
		av_score_tracker_tbd3do = AverageScoreTracker(files.shape, 'tbd3do', False)
	if do_sota18:
		av_score_tracker_sota18 = AverageScoreTracker(files.shape, 'sota18', False)
	if do_deblurgan:
		av_score_tracker_dg = AverageScoreTracker(files.shape, 'deblurgan', False)

	for kkf, ff in enumerate(files):
		if mode == 'tbd':
			if 'GX' in ff:
				medn = 7
			else:
				medn = 50
		gtp = GroundTruthProcessor(ff,kkf,folder,medn,shift,update_bg)
		if do_defmo:
			logger = SequenceLogger(log_folder, gtp)
		if do_deblatting:
			logger_tbd = SequenceLogger(log_folder, gtp, 'tbd')
			logger_tbd3d = SequenceLogger(log_folder, gtp, 'tbd3d')
			seq_score_tracker_tbd = SequenceScoreTracker(gtp.nfrms, 'tbd', False)
			seq_score_tracker_tbd3d = SequenceScoreTracker(gtp.nfrms, 'tbd3d', False)
		if do_sota18:
			seq_score_tracker_sota18 = SequenceScoreTracker(gtp.nfrms, 'sota18', False)
		if do_tbdo:
			logger_tbd3do = SequenceLogger(log_folder, gtp, 'tbd3do')
			seq_score_tracker_tbd3do = SequenceScoreTracker(gtp.nfrms, 'tbd3do', False)
		if do_deblurgan:
			seq_score_tracker_dg = SequenceScoreTracker(gtp.nfrms, 'deblurgan', False)

		seq_score_tracker = SequenceScoreTracker(gtp.nfrms)
		est_traj = None
		for kk in range(gtp.nfrms):
			gt_traj, radius, bbox = gtp.get_trajgt(kk)
			I, B = gtp.get_img(kk)
			gt_hs = gtp.get_hs(kk)

			if not gtp.w_trajgt:
				if gtp.use_hs:
					bbox, radius = fmo_detect_hs(gt_hs,B)
				else:
					bbox, radius = fmo_detect_maxarea(I,B)
				if np.min(radius) < 5:
					if verbose:
						print('Seq {}, frm {}, nothing found'.format(gtp.seqname, gtp.start_ind+kk))
					continue
			else:
				bbox = extend_bbox_uniform(bbox,radius,I.shape)

			bbox_tight = extend_bbox_uniform(bbox.copy(),10,I.shape)
			if gtp.use_hs:
				bbox_tight = bbox_fmo(bbox_tight,gt_hs,B)
			bbox = extend_bbox(bbox_tight.copy(),ext_factor*np.max(radius),g_resolution_y/g_resolution_x,I.shape)

			im_crop = crop_resize(I, bbox, (g_resolution_x, g_resolution_y))
			bgr_crop = crop_resize(B, bbox, (g_resolution_x, g_resolution_y))
			
			if do_defmo:
				preprocess = get_transform()
				input_batch = torch.cat((preprocess(im_crop), preprocess(bgr_crop)), 0).to(device).unsqueeze(0).float()
				with torch.no_grad():
					start = time.time()
					latent = encoder(input_batch)
					steps = gtp.nsplits*multi_f
					if eval_d > 1:
						steps = gtp.nsplits*eval_d + 1
						multi_f = 1
					times = torch.linspace(0,1,steps).to(device)
					renders_orig = rendering(latent,times[None])
					if eval_d > 1:
						renders = renders_orig[:,:-1].reshape(1, gtp.nsplits, eval_d, 4, g_resolution_y, g_resolution_x).mean(2)
					else:
						renders = renders_orig
					av_score_tracker.next_time(time.time() - start)

				renders_rgba = renders[0].data.cpu().detach().numpy().transpose(2,3,1,0)
				est_hs_crop = rgba2hs(renders_rgba, bgr_crop)

				est_traj_prev = est_traj
				if True:
					est_traj = renders2traj(renders,device)[0].T.cpu()
				else:
					est_traj = renders2traj_bbox(renders_orig[0,g_eval_d//2+1::g_eval_d].data.cpu().detach().numpy().transpose(2,3,1,0))
				est_traj = rev_crop_resize_traj(est_traj, bbox, (g_resolution_x, g_resolution_y))

			if do_deblatting or do_tbdo:
				if gtp.use_hs:
					bbox_temp = bbox_detect_hs(crop_only(gt_hs[:,:,:,0],bbox_tight), crop_only(B,bbox_tight))
					if len(bbox_temp) == 0:
						bbox_temp = bbox_tight
					debl_dim = bbox_temp[2:] - bbox_temp[:2]
				else:
					debl_dim = (radius,radius)
				bbox_debl = extend_bbox_uniform(bbox_tight.copy(),0.5*radius,I.shape)
				if do_tbdo and gtp.w_trajgt:
					rgba_tbd3d_or, Hso_crop = deblatting_oracle_runner(crop_only(I,bbox_debl),crop_only(B,bbox_debl),debl_dim,gt_traj[[1,0]]-bbox_debl[:2,None])
					Hso = rev_crop_resize(Hso_crop[:,:,None,:][:,:,[-1,-1,-1],:],bbox_debl,np.zeros(I.shape))
					est_hs_tbd3d0 = np.zeros(I.shape+(gtp.nsplits,))
					for tmki in range(gtp.nsplits): 
						Hsc = Hso[:,:,0,tmki]/np.sum(Hso[:,:,0,tmki])
						est_hs_tbd3d0[:,:,:,tmki] = fmo_model(B,Hsc,rgba_tbd3d_or[:,:,:3,tmki],rgba_tbd3d_or[:,:,3,tmki])
					if gtp.use_hs:
						seq_score_tracker_tbd3do.next_appearance(kk,crop_only(gt_hs,bbox_tight),crop_only(est_hs_tbd3d0,bbox_tight),crop_only(I,bbox_tight),crop_only(B,bbox_tight))
			
				if do_deblatting:
					start = time.time()
					est_hs_tbd_crop, est_hs_tbd3d_crop, rgba_tbd_crop, rgba_tbd3d_crop, est_traj_tbd, Hs = deblatting_runner(crop_only(I,bbox_debl),crop_only(B,bbox_debl),gtp.nsplits,debl_dim)
					av_score_tracker_tbd3d.next_time(time.time() - start)
					est_traj_tbd[0] += bbox_debl[1]
					est_traj_tbd[1] += bbox_debl[0]
					if gtp.use_hs:
						gt_hs_debl_crop = crop_only(gt_hs, bbox_debl)
						est_hs_tbd_crop, do_flip_debl = sync_directions(est_hs_tbd_crop, gt_hs_debl_crop)
						est_hs_tbd3d_crop, do_flip_debl = sync_directions(est_hs_tbd3d_crop, gt_hs_debl_crop)
						if do_flip_debl:
							rgba_tbd_crop = rgba_tbd_crop[:,:,:,::-1]
							rgba_tbd3d_crop = rgba_tbd3d_crop[:,:,:,::-1]
					est_hs_tbd = rev_crop_resize(est_hs_tbd_crop,bbox_debl,I)
					est_hs_tbd3d = rev_crop_resize(est_hs_tbd3d_crop,bbox_debl,I)
					rgb_tbd_crop = rev_crop_resize(rgba2rgb(rgba_tbd_crop),bbox_debl,I)
					rgb_tbd3d_crop = rev_crop_resize(rgba2rgb(rgba_tbd3d_crop),bbox_debl,I)
					logger_tbd.write_trajest(est_traj_tbd)
					if gtp.w_trajgt:
						iou = seq_score_tracker_tbd.next_traj(kk,gt_traj,est_traj_tbd,radius)
					if gtp.use_hs:
						seq_score_tracker_tbd.next_appearance(kk,crop_only(gt_hs,bbox_tight),crop_only(est_hs_tbd,bbox_tight),crop_only(I,bbox_tight),crop_only(B,bbox_tight))
						seq_score_tracker_tbd3d.next_appearance(kk,crop_only(gt_hs,bbox_tight),crop_only(est_hs_tbd3d,bbox_tight),crop_only(I,bbox_tight),crop_only(B,bbox_tight))
					logger_tbd3d.write_crops(kk,crop_only(rgb_tbd3d_crop,bbox_tight), crop_only(est_hs_tbd3d,bbox_tight),crop_only(gt_hs,bbox_tight),crop_only(I,bbox_tight),crop_only(B,bbox_tight))
					
			if do_sota18:
				if False:
					est_hs_sota18 = run_sota18(I)
				else:
					est_hs_sota18_crop = run_sota18(im_crop)
					est_hs_sota18 = rev_crop_resize(est_hs_sota18_crop,bbox,I)
				if gtp.use_hs:
					while est_hs_sota18.shape[3] < gt_hs.shape[3]:
						est_hs_sota18 = np.concatenate([est_hs_sota18, est_hs_sota18[...,-1:]],3)
					seq_score_tracker_sota18.next_appearance(kk,crop_only(gt_hs,bbox_tight),crop_only(est_hs_sota18,bbox_tight),crop_only(I,bbox_tight),crop_only(B,bbox_tight))
				else:
					est_hs_sota18 = np.concatenate([est_hs_sota18, est_hs_sota18[...,-1:]],3)
					
			if do_deblurgan:
				est_hs_deblurgan_crop = run_deblurgan(im_crop)
				est_hs_deblurgan = rev_crop_resize(est_hs_deblurgan_crop[...,None],bbox,I)[...,0]
				est_hs_deblurgan = np.repeat(est_hs_deblurgan[...,None],gtp.nsplits,3)
				if gtp.use_hs:
					seq_score_tracker_dg.next_appearance(kk,crop_only(gt_hs,bbox_tight),crop_only(est_hs_deblurgan,bbox_tight),crop_only(I,bbox_tight),crop_only(B,bbox_tight))
			
			if do_defmo:
				if multi_f > 1:
					gt_hs = np.repeat(gt_hs,multi_f,3)

				if gtp.w_trajgt:
					iou = seq_score_tracker.next_traj(kk,gt_traj,est_traj[:,::multi_f],radius)
					logger.write_trajgt(gt_traj)
				
				gt_hs_crop = crop_resize(gt_hs, bbox, (g_resolution_x, g_resolution_y))
				if gtp.use_hs:
					est_hs_crop, do_flip = sync_directions(est_hs_crop, gt_hs_crop)
				else:
					est_hs_crop, est_traj, do_flip = sync_directions_smooth(est_hs_crop, est_traj, est_traj_prev, radius)
				if do_flip:
					renders_rgba = renders_rgba[:,:,:,::-1]

				# est_hs = None
				# if renders[:,:,-1:].max() > 0.05:
				logger.write_trajest(est_traj)
				est_hs = rev_crop_resize(est_hs_crop,bbox,I)
				if gtp.use_hs:
					# logger.write_crops(kk,crop_only(rgba2rgb(rev_crop_resize(renders_rgba,bbox,np.ones(I.shape[:2]+(4,)))),bbox_tight), crop_only(est_hs,bbox_tight),crop_only(gt_hs,bbox_tight),crop_only(I,bbox_tight),crop_only(B,bbox_tight))
					logger.write_crops_3c(kk,crop_only(rgba2rgb(rev_crop_resize(renders_rgba,bbox,np.ones(I.shape[:2]+(4,)))),bbox_tight), crop_only(est_hs,bbox_tight),crop_only(gt_hs,bbox_tight),crop_only(I,bbox_tight),crop_only(B,bbox_tight))
					seq_score_tracker.next_appearance(kk,crop_only(gt_hs,bbox_tight),crop_only(est_hs,bbox_tight),crop_only(I,bbox_tight),crop_only(B,bbox_tight))
					if eval_gt:
						input_gt_batch = torch.zeros((gtp.nsplits,6,)+gt_hs_crop.shape[:2]).to(device).float()
						for tempi in range(gtp.nsplits):
							input_gt_batch[tempi] = torch.cat((preprocess(gt_hs_crop[:,:,:,tempi*multi_f]), preprocess(bgr_crop)), 0).to(device).float()
						with torch.no_grad():
							latent = encoder(input_gt_batch)
							times = torch.linspace(0,1,eval_gt_steps+1)[:-1].to(device)
							renders_gt = rendering(latent,times[None].repeat(gtp.nsplits,1))
						renders_gt_rgba = renders_gt.data.cpu().detach().numpy().transpose(3,4,2,0,1)
						est_gt_hs_crop = np.zeros(renders_gt_rgba.shape[:2]+(3,)+renders_gt_rgba.shape[3:])
						old_traj = None
						for tempi in range(gtp.nsplits):
							est_gt_traj = renders2traj(renders_gt[tempi:(tempi+1)],device)[0].T.cpu().detach().numpy()
							est_gt_hs_crop[:,:,:,tempi,:], nothing1, nothing2 = sync_directions_smooth(rgba2hs(renders_gt_rgba[:,:,:,tempi,:], bgr_crop), est_gt_traj, old_traj, 200)
							old_traj = est_gt_traj
						est_gt_hs = rev_crop_resize(est_gt_hs_crop.reshape(est_gt_hs_crop.shape[:3]+(eval_gt_steps*gtp.nsplits,)) ,bbox,I)
						logger.write_crops_4c(kk,crop_only(rgba2rgb(rev_crop_resize(renders_rgba,bbox,np.ones(I.shape[:2]+(4,)))),bbox_tight), crop_only(est_hs,bbox_tight),crop_only(gt_hs,bbox_tight),crop_only(I,bbox_tight),crop_only(B,bbox_tight),crop_only(est_gt_hs,bbox_tight))
							
				else:
					# logger.write_crops(kk,rgba2rgb(renders_rgba), est_hs_crop, gt_hs_crop, im_crop, bgr_crop)
					logger.write_crops_3c(kk,rgba2rgb(renders_rgba), est_hs_crop, gt_hs_crop, im_crop, bgr_crop)

				logger.write_superres(I,est_hs,gt_hs)
				if verbose:
					seq_score_tracker.report(gtp.seqname, kk)

			if do_deblatting:
				logger_tbd.write_superres(I,est_hs_tbd,gt_hs)
				logger_tbd3d.write_superres(I,est_hs_tbd3d,gt_hs)
				if verbose:
					seq_score_tracker_tbd.report(gtp.seqname, kk)
					seq_score_tracker_tbd3d.report(gtp.seqname, kk)
			if do_tbdo and verbose:
				logger_tbd3do.write_superres(I,est_hs_tbd3d0,gt_hs)
				seq_score_tracker_tbd3do.report(gtp.seqname, kk)
			if do_sota18 and verbose:
				seq_score_tracker_sota18.report(gtp.seqname, kk)
			if do_deblurgan and verbose:
				seq_score_tracker_dg.report(gtp.seqname, kk)

		if do_defmo:
			av_score_tracker.next(gtp.seqname, seq_score_tracker)
			logger.close()
		if do_deblatting:
			av_score_tracker_tbd.next(gtp.seqname, seq_score_tracker_tbd)
			av_score_tracker_tbd3d.next(gtp.seqname, seq_score_tracker_tbd3d)
			logger_tbd.close()
			logger_tbd3d.close()
		if do_tbdo:
			av_score_tracker_tbd3do.next(gtp.seqname, seq_score_tracker_tbd3do)
			logger_tbd3do.close()
		if do_sota18:
			av_score_tracker_sota18.next(gtp.seqname, seq_score_tracker_sota18)
		if do_deblurgan:
			av_score_tracker_dg.next(gtp.seqname, seq_score_tracker_dg)
	
	if do_defmo:
		av_score_tracker.close()
	if do_deblatting:
		av_score_tracker_tbd.close()
		av_score_tracker_tbd3d.close()
	if do_tbdo:
		av_score_tracker_tbd3do.close()
	if do_sota18:
		av_score_tracker_sota18.close()
	if do_deblurgan:
		av_score_tracker_dg.close()

	torch.cuda.empty_cache()
	# pdb.set_trace()


def get_tbd_sample(framenum = -5, fileind = -1, mode = 'tbd'):
	nmed = 7
	if mode == 'tbd':
		files = get_tbd_dataset()
		folder = g_tbd_folder
	elif mode == 'tbd3d':
		files = get_tbd3d_dataset()
		folder = g_tbd3d_folder
	elif mode == 'tbdfalling':
		files = get_falling_dataset()
		folder = g_falling_folder
	else:
		print('Mode not found!')

	aspect_ratio = g_resolution_y / g_resolution_x
	for ff in files[fileind:(fileind+1)]:
		seqname = os.path.split(ff)[-1][:-4]
		seqpath = folder + 'imgs/' + seqname + '/'
		if not os.path.exists(seqpath):
			print('Directory does not exist')
			return None

		path = seqpath + "{:08d}.png".format(framenum)
		Im0 = cv2.imread(path,cv2.IMREAD_UNCHANGED)/255
		nfrm = np.min([nmed,framenum+1])
		Vk = np.zeros(Im0.shape+(nfrm+nmed,))
		for kk in range(nfrm):
			Vk[:,:,:,kk] = cv2.imread(seqpath + "{:08d}.png".format(framenum-kk),cv2.IMREAD_UNCHANGED)/255
		for kk in range(nmed):
			Vk[:,:,:,nfrm+kk] = cv2.imread(seqpath + "{:08d}.png".format(framenum+kk+1),cv2.IMREAD_UNCHANGED)/255
		
		Vk = Vk[:,:,[2,1,0]]
		I = Vk[:,:,:,0]
		B = np.median(Vk,3)

		if mode == 'tbd3d' or mode == 'tbdfalling':
			bbox, minor_axis_length = fmo_detect_maxarea(I,B)
		elif mode == 'tbd':
			bbox, minor_axis_length = fmo_detect(I,B)

		bbox = np.array(bbox)
		height, width = bbox[2] - bbox[0], bbox[3] - bbox[1]
		
		h2 = height*2

		h2 = int(np.ceil(np.ceil(h2 / aspect_ratio) * aspect_ratio))
		w2 = int(h2 / aspect_ratio)

		wdiff = w2 - width
		wdiff2 = int(wdiff/2)
		hdiff = h2 - height
		hdiff2 = int(hdiff/2)

		bbox[0] -= hdiff2
		bbox[2] += hdiff-hdiff2
		bbox[1] -= wdiff2
		bbox[3] += wdiff-wdiff2

		im = I[bbox[0]:bbox[2], bbox[1]:bbox[3], :]
		imr = cv2.resize(im, (g_resolution_x, g_resolution_y), interpolation = cv2.INTER_CUBIC)

		bgr = B[bbox[0]:bbox[2], bbox[1]:bbox[3], :]
		bgrr = cv2.resize(bgr, (g_resolution_x, g_resolution_y), interpolation = cv2.INTER_CUBIC)
		preprocess = get_transform()
		input_batch = torch.cat((preprocess(imr), preprocess(bgrr)), 0).float()

		# pdb.set_trace()
		# save_image(preprocess(I).clone(),'/home.stud/rozumden/tmp.png')

		return input_batch


def dataset2png(mode = 'tbd'):
	generate_LS = False
	generate_HS = True
	kn = 8
	use_gt_bbox = True
	if mode == 'tbd':
		files = get_tbd_dataset()
		folder = g_tbd_folder
	elif mode == 'tbd3d':
		files = get_tbd3d_dataset()
		folder = g_tbd3d_folder
	elif mode == 'tbdfalling':
		files = get_falling_dataset()
		folder = g_falling_folder
	else:
		print('Mode not found!')

	# pdb.set_trace()
	seqi = 0
	for ff in files:
		new_version = True
		try:
			f = h5py.File(ff, 'r')
		except: ## older matlab versions
			new_version = False
			f = scipy.io.loadmat(ff)

		seqname = os.path.split(ff)[-1][:-4]
		seqpath = folder + 'imgs/' + seqname + '/'
		if not os.path.exists(seqpath):
			os.makedirs(seqpath)

		keys = f.keys()

		if generate_LS and 'Vk' in keys:
			if new_version:
				Vk = (np.array(f['Vk']).transpose(3,2,1,0)[:,:,:,:]/255).astype(np.float32) # [2,1,0]
			else:
				Vk = (np.array(f['Vk'])/255).astype(np.float32)
		else:
			ims = np.array(f['V'])
			if ims.shape[2] == 3:
				V = (ims.transpose(1,0,2,3)[::-1]/255).astype(np.float32)
				if generate_LS:
					Vk = generate_lowFPSvideo(V,k=kn,do_WB=False,gamma_coef=1.0).astype(np.float32)
			else:
				V = (ims.transpose(3,2,1,0)[:,:,:,:]/255).astype(np.float32)
				V = generate_lowFPSvideo(V,k=1).astype(np.float32)
				if generate_LS:
					Vk = generate_lowFPSvideo(V,k=kn).astype(np.float32)

		if generate_LS:
			pars = []
			if 'PAR' in keys:
				pars = f['PAR']
			elif 'POS' in keys:
				pars = f['POS']
			all_par = []
			for kk in range(Vk.shape[3]):
				if pars != []:
					if new_version:
						if 'PAR' in keys:
							par = np.array(f[pars[kk][0]]).T
						elif 'R' in keys:
							par = (np.array(pars[kk*kn:kn*(kk+1),:]).T)
							rad = (f['R'][:].squeeze()[kk*kn:kn*(kk+1)])
					else:
						par = pars[0][kk]
					if all_par == []:
						all_par = np.zeros((Vk.shape[3]*2, par.shape[1]))
						all_rad = np.zeros((Vk.shape[3], par.shape[1]))
					all_par[2*kk:2*kk+2,:] = par
					if 'R' in keys:
		 				all_rad[kk:kk+1,:] = rad
				I = Vk[:,:,:,kk]
				path = seqpath + "{:08d}.png".format(kk)
				Ims = Image.fromarray((I * 255).astype(np.uint8))
				Ims.save(path)
				print('Seq {}, frm {}'.format(seqname, kk))
			if pars != []:
				np.savetxt(seqpath + 'gt.txt', all_par, fmt='%.2f')
				if 'R' in keys:
					np.savetxt(seqpath + 'gtr.txt', all_rad, fmt='%.2f')
		if generate_HS:
			hspath = folder + 'imgs_gt/' + seqname + '/'
			if not os.path.exists(hspath):
				os.makedirs(hspath)
			for kk in range(V.shape[3]):
				I = V[:,:,:,kk]
				path = hspath + "{:08d}.png".format(kk)
				Ims = Image.fromarray((I * 255).astype(np.uint8))
				Ims.save(path)
				print('Seq {}, hs frm {}'.format(seqname, kk))

		if new_version:
			f.close()

		print('Finished seq {}'.format(seqname))
		seqi += 1
	return None


