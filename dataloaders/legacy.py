#### LEGACY

def get_tbd_sample_mat(framenum = -5, fileind = -1, mode = 'tbd'):
	if mode == 'tbd':
		files = get_tbd_dataset()
	elif mode == 'tbd3d':
		files = get_tbd3d_dataset()
	elif mode == 'tbdfalling':
		files = get_falling_dataset()
	else:
		print('Mode not found!')

	aspect_ratio = g_resolution_y / g_resolution_x
	for ff in files[fileind:(fileind+1)]:
		if mode == 'tbdfalling':
			f = scipy.io.loadmat(ff)
		else:
			f = h5py.File(ff, 'r')

		keys = f.keys()

		if 'Vk' in keys:
			Vk = (np.array(f['Vk']).transpose(3,2,1,0)[:,:,:,:]/255).astype(np.float32) # [2,1,0]
		else:
			ims = np.array(f['V'])
			if ims.shape[2] == 3:
				V = (ims.transpose(1,0,2,3)[::-1]/255).astype(np.float32)
				Vk = generate_lowFPSvideo(V,k=8,do_WB=False,gamma_coef=1.0).astype(np.float32)
			else:
				V = (ims.transpose(3,2,1,0)[:,:,:,:]/255).astype(np.float32)
				Vk = generate_lowFPSvideo(V,k=8).astype(np.float32)

		if mode != 'tbdfalling':
			f.close()

		I = Vk[:,:,:,framenum]
		B = np.median(Vk[:,:,:,max(framenum-5,0):(framenum+5)],3)
		# B = np.median(Vk,3)

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
		input_batch = torch.cat((preprocess(imr), preprocess(bgrr)), 0)

		# pdb.set_trace()
		# save_image(preprocess(I).clone(),'/home.stud/rozumden/tmp.png')

		return input_batch

def evaluate_on_mat(encoder, rendering, device, mode = 'tbd'):
	log_folder = '/home.stud/rozumden/tmp/'+mode+'_eval/'
	medn = 5
	use_gt_bbox = True
	if mode == 'tbd':
		files = get_tbd_dataset()
	elif mode == 'tbd3d':
		files = get_tbd3d_dataset()
	elif mode == 'tbdfalling':
		files = get_falling_dataset()
	else:
		print('Mode not found!')

	aspect_ratio = g_resolution_y / g_resolution_x
	av_ious = np.zeros(files.shape)
	seqi = 0
	for ff in files:
		if mode == 'tbdfalling':
			f = scipy.io.loadmat(ff)
		else:
			f = h5py.File(ff, 'r')

		seqname = os.path.split(ff)[-1][:-4]
		seqpath = log_folder + seqname + '/'
		if not os.path.exists(seqpath):
			os.makedirs(seqpath)

		keys = f.keys()

		if 'Vk' in keys:
			Vk = (np.array(f['Vk']).transpose(3,2,1,0)[:,:,:,:]/255).astype(np.float32) # [2,1,0]
		else:
			ims = np.array(f['V'])
			if ims.shape[2] == 3:
				V = (ims.transpose(1,0,2,3)[::-1]/255).astype(np.float32)
				Vk = generate_lowFPSvideo(V,k=8,do_WB=False,gamma_coef=1.0).astype(np.float32)
			else:
				V = (ims.transpose(3,2,1,0)[:,:,:,:]/255).astype(np.float32)
				Vk = generate_lowFPSvideo(V,k=8).astype(np.float32)

		all_ious = np.zeros(Vk.shape[-1])
		pars = f['PAR']
		B = np.median(Vk[:,:,:,:medn],3)

		ImGT = B.copy()
		ImEst = B.copy()

		for kk in range(Vk.shape[3]):
			par = np.array(f[pars[kk][0]])
			I = Vk[:,:,:,kk]
			if kk >= medn:
				B = np.median(Vk[:,:,:,(kk-medn+1):kk+1],3)

			if use_gt_bbox:
				pdb.set_trace()
				bbox = [par.min(0), par.max(0)]
			else:
				if mode == 'tbd3d' or mode == 'tbdfalling':
					bbox, minor_axis_length = fmo_detect_maxarea(I,B)
				elif mode == 'tbd':
					bbox, minor_axis_length = fmo_detect(I,B)

			if minor_axis_length < 1:
				print('Seq {}, frm {}, iou nothing found'.format(seqname, kk))
				continue

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
			input_batch = torch.cat((preprocess(imr), preprocess(bgrr)), 0).to(device).unsqueeze(0)
			with torch.no_grad():
				latent = encoder(input_batch)
				steps = par.shape[0]
				times = torch.linspace(0,1,steps).to(device)
				renders = rendering(latent,times[None])

			rad = np.round(minor_axis_length)
			gt_traj = par.T
			# pdb.set_trace()

			est_traj = renders2traj(renders,device)[0].T
			est_traj *= (im.shape[0]/imr.shape[0])
			est_traj[0] += bbox[0]
			est_traj[1] += bbox[1]
			est_traj = np.array(est_traj[[1,0]].cpu())

			ious = calciou(gt_traj, est_traj, rad)
			ious2 = calciou(gt_traj, est_traj[:,-1::-1], rad)
			iou = np.max([np.mean(ious), np.mean(ious2)])
			print('Seq {}, frm {}, iou {}'.format(seqname, kk, iou))

			ImGT[gt_traj[1].astype(int),gt_traj[0].astype(int),1] = 1.0
			save_image(preprocess(ImGT).clone(),seqpath + 'imgt.png')
			ImEst[est_traj[1].astype(int),est_traj[0].astype(int),0] = 1.0
			save_image(preprocess(ImEst).clone(),seqpath + 'imest.png')

			if iou > np.max(all_ious):
				save_image(preprocess(imr).clone(),seqpath+'bestim.png')
				write_latent(rendering, latent, device, folder=seqpath, steps=24)

			all_ious[kk] = iou

		if mode != 'tbdfalling':
			f.close()

		av_ious[seqi] = np.mean(all_ious[all_ious > 0])
		print('Finished seq {}, average tiou {}'.format(seqname, av_ious[seqi]))
		seqi += 1

	# pdb.set_trace()