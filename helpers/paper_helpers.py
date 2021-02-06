import torch
from dataloaders.reporters import *
from dataloaders.loader import *
from dataloaders.tbd_loader import *
from main_settings import *
from utils import *
from scipy.ndimage.morphology import binary_dilation
import os
import sys
import pdb

def get_figure_images(encoder, rendering, device, mode, seqi, frmi, results_mode = False, n_occ=2):
    log_folder = './output/'
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    medn = 7
    gamma_c = 1 # 0.7
    ext_factor = 4
    multi_f = 1

    do_deblatting = False
    do_sota18 = False
    do_deblurgan = False

    do_tbdo = False

    if do_sota18:
        from helpers.sota18_runner import run_sota18
    if do_deblurgan:
        sys.path.insert(0, './DeblurGANv2')
        from helpers.deblurgan_runner import run_deblurgan
        
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
        medn = 50
        ext_factor = 4
    elif mode == 'wildfmo':
        files = get_wildfmo_dataset()
        folder = g_wildfmo_folder
        medn = 50
        ext_factor = 1
    elif mode == 'train':
        files = g_render_objs_train
        folder = g_dataset_folder
        ext_factor = 1
    elif mode == 'val':
        files = g_render_objs_train
        folder = g_validation_folder
        ext_factor = 1
    elif mode == 'youtube':
        files = get_youtube_dataset()
        folder = g_youtube_folder
        ext_factor = 4
        medn = 9
    else:
        print('Mode not found!')

    for kkf, ff in enumerate(files):
        isdigit = str(seqi).isdigit()
        if (isdigit and kkf != seqi) or (not isdigit and not seqi in ff):
            continue

        gtp = GroundTruthProcessor(ff,kkf,folder,medn,0)
        do_tbdo = do_tbdo and gtp.w_trajgt

        seq_score_tracker = SequenceScoreTracker(gtp.nfrms)
        seq_score_tracker_tbd = SequenceScoreTracker(gtp.nfrms, 'tbd', False)
        seq_score_tracker_tbd3d = SequenceScoreTracker(gtp.nfrms, 'tbd3d', False)

        est_traj = None
        for kk in range(gtp.nfrms):
            I, B = gtp.get_img(kk)
            if kk != frmi:
                continue

            gt_traj, radius, bbox = gtp.get_trajgt(kk)
            gt_hs = gtp.get_hs(kk)

            if gtp.use_hs:
                gt_hs[gt_hs<0] = 0

            if not gtp.w_trajgt:
                if gtp.use_hs:
                    bbox, radius = fmo_detect_hs(gt_hs,B)
                else:
                    bbox, radius = fmo_detect_maxarea(I,B)
            else:
                bbox = extend_bbox_uniform(bbox,radius,I.shape)

            bbox_tight = extend_bbox_uniform(bbox.copy(),10,I.shape)
            if gtp.use_hs and not gtp.syn_mode:
                bbox_tight = bbox_fmo(bbox_tight,gt_hs,B)

            if n_occ > 9:
                bbox_tight = extend_bbox_uniform(bbox_tight.copy(),3,I.shape)
                ext2 = ((bbox_tight[2] - bbox_tight[0]) - (bbox_tight[3] - bbox_tight[1]))//2 + (bbox_tight[2] - bbox_tight[0])//4
                if ext2 > 0:
                    bbox_tight = extend_bbox_nonuniform(bbox_tight.copy(),[0,ext2],I.shape)
            else:
                bbox_tight = extend_bbox_uniform(bbox_tight.copy(),3,I.shape)

            bbox = extend_bbox(bbox_tight.copy(),ext_factor*np.max(radius),g_resolution_y/g_resolution_x,I.shape)
            
            im_crop = crop_resize(I, bbox, (g_resolution_x, g_resolution_y))
            bgr_crop = crop_resize(B, bbox, (g_resolution_x, g_resolution_y))

            eps = 0
            preprocess = get_transform()
            input_batch = torch.cat((preprocess(im_crop), preprocess(bgr_crop)), 0).to(device).unsqueeze(0).float()
            with torch.no_grad():
                start = time.time()
                latent = encoder(input_batch)
                steps = gtp.nsplits*multi_f
                times = torch.linspace(0+eps,1-eps,steps).to(device)
                renders = rendering(latent,times[None])

            # start = time.time(); latent = encoder(input_batch); times = torch.linspace(0,1,2).to(device); renders = rendering(latent,times[None]); end = time.time(); print( 1/(end - start) )
            # start = time.time(); rgba_tbd3d_or, Hso_crop = deblatting_oracle_runner(crop_only(I,bbox),crop_only(B,bbox),debl_dim,gt_traj[[1,0]]-bbox[:2,None]); end = time.time(); print( 1/(end - start) )
            # start = time.time(); est_hs_tbd_crop, est_hs_tbd3d_crop, rgba_tbd, rgba_tbd3d, est_traj_tbd, Hs_crop = deblatting_runner(crop_only(I,bbox),crop_only(B,bbox),gtp.nsplits,debl_dim); end = time.time(); print( 1/(end - start) )
            # start = time.time(); H,F,M = deblatting_single_runner(crop_only(I,bbox),crop_only(B,bbox),gtp.nsplits,debl_dim); end = time.time(); print( 1/(end - start) )
            
            renders_rgba = renders[0].data.cpu().detach().numpy().transpose(2,3,1,0)
            est_hs_crop = rgba2hs(renders_rgba, bgr_crop)
           
            est_traj_prev = est_traj
            if False:
                est_traj = renders2traj(renders,device)[0].T.cpu()
            else:
                est_traj = renders2traj_bbox(renders_rgba)
            est_traj = rev_crop_resize_traj(est_traj, bbox, (g_resolution_x, g_resolution_y))

            if do_deblatting or do_tbdo:
                if gtp.syn_mode:
                    bbox_temp = bbox_detect_hs(crop_only(gt_hs[:,:,[3,3,3],-1],bbox_tight), crop_only(np.zeros(B.shape),bbox_tight))
                    debl_dim = bbox_temp[2:] - bbox_temp[:2] + np.round(0.2*radius).astype(bbox_temp.dtype)
                elif gtp.use_hs:
                    bbox_temp = bbox_detect_hs(crop_only(gt_hs[:,:,:3,-1],bbox_tight), crop_only(B,bbox_tight))
                    if len(bbox_temp) == 0:
                        bbox_temp = bbox_tight
                    debl_dim = bbox_temp[2:] - bbox_temp[:2] + np.round(0.2*radius).astype(bbox_temp.dtype)
                else:
                    # debl_dim = (radius,radius)
                    debl_dim = (int(radius),int(radius))
                bbox_debl = extend_bbox_uniform(bbox_tight.copy(),0.5*radius,I.shape)
                start = time.time()
                if do_tbdo and gtp.w_trajgt:
                    rgba_tbd3d_or, Hso_crop = deblatting_oracle_runner(crop_only(I,bbox_debl),crop_only(B,bbox_debl),debl_dim,gt_traj[[1,0]]-bbox_debl[:2,None])
                    Hso = rev_crop_resize(Hso_crop[:,:,None,:][:,:,[-1,-1,-1],:],bbox_debl,np.zeros(I.shape))
               
                if do_deblatting:
                    est_hs_tbd_crop, est_hs_tbd3d_crop, rgba_tbd, rgba_tbd3d, est_traj_tbd, Hs_crop = deblatting_runner(crop_only(I,bbox_debl),crop_only(B,bbox_debl),gtp.nsplits,debl_dim)
                    est_traj_tbd[0] += bbox_debl[1]
                    est_traj_tbd[1] += bbox_debl[0]
                    if gtp.use_hs:
                        gt_hs_debl_crop = crop_only(gt_hs, bbox_debl)[:,:,:3]
                        est_hs_tbd_crop, do_flip_debl = sync_directions(est_hs_tbd_crop, gt_hs_debl_crop)
                        est_hs_tbd3d_crop, do_flip_debl = sync_directions(est_hs_tbd3d_crop, gt_hs_debl_crop)
                        if do_flip_debl:
                            rgba_tbd = rgba_tbd[:,:,:,::-1]
                            rgba_tbd3d = rgba_tbd3d[:,:,:,::-1]
                    est_hs_tbd = rev_crop_resize(est_hs_tbd_crop,bbox_debl,I)
                    est_hs_tbd3d = rev_crop_resize(est_hs_tbd3d_crop,bbox_debl,I)
                    Hs = rev_crop_resize(Hs_crop[:,:,None,:][:,:,[-1,-1,-1],:],bbox_debl,np.zeros(I.shape))
               
            if do_sota18:
                # est_hs_sota18 = run_sota18(I)
                est_hs_sota18_crop = run_sota18(im_crop)
                est_hs_sota18 = rev_crop_resize(est_hs_sota18_crop,bbox,I)

            if do_deblurgan:
                # est_hs_deblurgan = run_deblurgan(I)
                est_hs_deblurgan_crop = run_deblurgan(im_crop)
                est_hs_deblurgan_hr = rev_crop_resize(est_hs_deblurgan_crop[...,None],bbox,I)[...,0]

            if gtp.w_trajgt:
                iou = seq_score_tracker.next_traj(kk,gt_traj,est_traj,radius)

            if gtp.use_hs:
                gt_hs_crop = crop_resize(gt_hs[:,:,:3], bbox, (g_resolution_x, g_resolution_y))
                est_hs_crop, do_flip = sync_directions(est_hs_crop, gt_hs_crop)
            else:
                est_hs_crop, est_traj, do_flip = sync_directions_smooth(est_hs_crop, est_traj, est_traj_prev, radius)
            if do_flip:
                renders_rgba = renders_rgba[:,:,:,::-1]

            est_hs = rev_crop_resize(est_hs_crop,bbox,I)
            est_hs[est_hs<0]=0
           
            seq_score_tracker.report(gtp.seqname, kk)
            if do_deblatting: # and gtp.use_hs
                seq_score_tracker_tbd.next_appearance(kk,crop_only(gt_hs[:,:,:3],bbox_tight),crop_only(est_hs_tbd,bbox_tight),crop_only(I,bbox_tight),crop_only(B,bbox_tight))
                seq_score_tracker_tbd3d.next_appearance(kk,crop_only(gt_hs[:,:,:3],bbox_tight),crop_only(est_hs_tbd3d,bbox_tight),crop_only(I,bbox_tight),crop_only(B,bbox_tight))
                seq_score_tracker_tbd.report(gtp.seqname, kk)
                seq_score_tracker_tbd3d.report(gtp.seqname, kk)
                
            # pdb.set_trace()
            renders_rgba_full = rev_crop_resize(renders_rgba,bbox,np.zeros(I.shape[:2]+(4,)))
            # inds = [0,4,gtp.nsplits-1]
            inds0 = np.array([0,4])
            offsets = np.array([0,3])
            if results_mode:
                if n_occ == 2:
                    inds0 = np.array([0])
                    offsets = np.array([0,7])
                else:
                    inds0 = np.array([0])
                    offsets = np.array(range(gtp.nsplits))

            for tempi in offsets:
                inds = inds0 + tempi
                renders_a_joint = renders_rgba_full[:,:,:,inds].sum(3)
                est_hs_joint = B
                renders_rgba_joint = (0.3**(1/gamma_c))*np.ones(B.shape)
                if gtp.syn_mode:
                    gt_hs_sum = np.ones(B.shape)
                    gt_hs_a = gt_hs[:,:,3:,inds].sum(3)
                    renders_rgba_joint = np.ones(B.shape)
                elif gtp.use_hs:
                    gt_hs_sum = gt_hs[:,:,:3,inds[0]]

                if do_deblatting:
                    renders_a_joint_tbd = np.zeros(B.shape)
                    renders_a_joint_tbd3d = renders_a_joint_tbd
                    est_hs_joint_tbd = est_hs_joint
                    renders_rgba_joint_tbd = renders_rgba_joint
                    est_hs_joint_tbd3d = est_hs_joint
                    renders_rgba_joint_tbd3d = renders_rgba_joint
                if do_tbdo:
                    renders_a_joint_tbd3do = np.zeros(B.shape)
                    est_hs_joint_tbd3do = est_hs_joint
                    renders_rgba_joint_tbd3do = renders_rgba_joint

                for ind0 in inds:
                    est_hs_joint = rgba2hs(renders_rgba_full[...,ind0:(ind0+1)], est_hs_joint)[...,0]
                    renders_rgba_joint = rgba2hs(renders_rgba_full[...,ind0:(ind0+1)], renders_rgba_joint)[...,0]
                    mask = binary_dilation(renders_rgba_full[:,:,-1,ind0] > 0.001, iterations=10)[:,:,None][:,:,[0,0,0]]
                    if gtp.syn_mode:
                        gt_hs_sum = rgba2hs(gt_hs[...,ind0:(ind0+1)], gt_hs_sum)[...,0]
                    elif gtp.use_hs:
                        if inds[0] != ind0:
                            gt_hs_sum[mask] = gt_hs[:,:,:3,ind0][mask]

                    if do_deblatting:
                        Hsc = Hs[:,:,0,ind0]/np.sum(Hs[:,:,0,ind0])
                        renders_a_joint_tbd = fmo_model(renders_a_joint_tbd,Hsc,rgba_tbd[:,:,[3,3,3],ind0],rgba_tbd[:,:,3,ind0])
                        est_hs_joint_tbd = fmo_model(est_hs_joint_tbd,Hsc,rgba_tbd[:,:,:3,ind0],rgba_tbd[:,:,3,ind0])
                        renders_rgba_joint_tbd = fmo_model(renders_rgba_joint_tbd,Hsc,rgba_tbd[:,:,:3,ind0],rgba_tbd[:,:,3,ind0])

                        renders_a_joint_tbd3d = fmo_model(renders_a_joint_tbd3d,Hsc,rgba_tbd3d[:,:,[3,3,3],ind0],rgba_tbd3d[:,:,3,ind0])
                        est_hs_joint_tbd3d = fmo_model(est_hs_joint_tbd3d,Hsc,rgba_tbd3d[:,:,:3,ind0],rgba_tbd3d[:,:,3,ind0])
                        renders_rgba_joint_tbd3d = fmo_model(renders_rgba_joint_tbd3d,Hsc,rgba_tbd3d[:,:,:3,ind0],rgba_tbd3d[:,:,3,ind0])
                    if do_tbdo:
                        Hsco = Hso[:,:,0,ind0]/np.sum(Hso[:,:,0,ind0])
                        renders_a_joint_tbd3do = fmo_model(renders_a_joint_tbd3do,Hsco,rgba_tbd3d_or[:,:,[3,3,3],ind0],rgba_tbd3d_or[:,:,3,ind0])
                        est_hs_joint_tbd3do = fmo_model(est_hs_joint_tbd3do,Hsco,rgba_tbd3d_or[:,:,:3,ind0],rgba_tbd3d_or[:,:,3,ind0])
                        renders_rgba_joint_tbd3do = fmo_model(renders_rgba_joint_tbd3do,Hsco,rgba_tbd3d_or[:,:,:3,ind0],rgba_tbd3d_or[:,:,3,ind0])
                    

                est_hs_joint[est_hs_joint<0]=0
                renders_rgba_joint[renders_rgba_joint<0]=0
                imwrite(crop_only(1-renders_a_joint[...,[-1,-1,-1]],bbox_tight),log_folder+gtp.seqname+'_{:04d}_estm{}.png'.format(kk,tempi))
                imwrite(crop_only(renders_rgba_joint,bbox_tight)**gamma_c,log_folder+gtp.seqname+'_{:04d}_estfm{}.png'.format(kk,tempi))
                imwrite(crop_only(est_hs_joint,bbox_tight)**gamma_c,log_folder+gtp.seqname+'_{:04d}_est{}.png'.format(kk,tempi))
                if do_deblatting:
                    renders_a_joint_tbd[renders_a_joint_tbd<0]=0
                    renders_a_joint_tbd3d[renders_a_joint_tbd3d<0]=0
                    est_hs_joint_tbd[est_hs_joint_tbd<0]=0
                    renders_rgba_joint_tbd[renders_rgba_joint_tbd<0]=0
                    est_hs_joint_tbd3d[est_hs_joint_tbd3d<0]=0
                    renders_rgba_joint_tbd3d[renders_rgba_joint_tbd3d<0]=0
                    imwrite(crop_only(1-renders_a_joint_tbd[...,[-1,-1,-1]],bbox_tight),log_folder+gtp.seqname+'_{:04d}_tbdm{}.png'.format(kk,tempi))
                    imwrite(crop_only(1-renders_a_joint_tbd3d[...,[-1,-1,-1]],bbox_tight),log_folder+gtp.seqname+'_{:04d}_tbd3dm{}.png'.format(kk,tempi))
                    imwrite(crop_only(renders_rgba_joint_tbd,bbox_tight)**gamma_c,log_folder+gtp.seqname+'_{:04d}_tbdfm{}.png'.format(kk,tempi))
                    imwrite(crop_only(renders_rgba_joint_tbd3d,bbox_tight)**gamma_c,log_folder+gtp.seqname+'_{:04d}_tbd3dfm{}.png'.format(kk,tempi))
                    imwrite(crop_only(est_hs_joint_tbd,bbox_tight)**gamma_c,log_folder+gtp.seqname+'_{:04d}_tbd{}.png'.format(kk,tempi))
                    imwrite(crop_only(est_hs_joint_tbd3d,bbox_tight)**gamma_c,log_folder+gtp.seqname+'_{:04d}_tbd3d{}.png'.format(kk,tempi))

                if gtp.syn_mode:
                    imwrite(crop_only(1-gt_hs_a[...,[-1,-1,-1]],bbox_tight)**gamma_c,log_folder+gtp.seqname+'_{:04d}_hsm{}.png'.format(kk,tempi))
                    imwrite(crop_only(gt_hs_sum,bbox_tight)**gamma_c,log_folder+gtp.seqname+'_{:04d}_hs{}.png'.format(kk,tempi))
                elif gtp.use_hs:
                    gt_hs_a = np.sum(np.abs(gt_hs - B[...,None]),2)[:,:,None,:].astype(float)*8
                    imwrite(crop_only(1-gt_hs_a[...,[-1,-1,-1],inds[0]],bbox_tight)**gamma_c,log_folder+gtp.seqname+'_{:04d}_hsm{}.png'.format(kk,tempi))
                    imwrite(crop_only(gt_hs_sum,bbox_tight)**gamma_c,log_folder+gtp.seqname+'_{:04d}_hs{}.png'.format(kk,tempi))
               

                if do_tbdo:
                    renders_a_joint_tbd3do[renders_a_joint_tbd3do<0]=0
                    est_hs_joint_tbd3do[est_hs_joint_tbd3do<0]=0
                    renders_rgba_joint_tbd3do[renders_rgba_joint_tbd3do<0]=0
                    imwrite(crop_only(1-renders_a_joint_tbd3do[...,[-1,-1,-1]],bbox_tight),log_folder+gtp.seqname+'_{:04d}_tbd3dom{}.png'.format(kk,tempi))
                    imwrite(crop_only(renders_rgba_joint_tbd3do,bbox_tight)**gamma_c,log_folder+gtp.seqname+'_{:04d}_tbd3dofm{}.png'.format(kk,tempi))
                    imwrite(crop_only(est_hs_joint_tbd3do,bbox_tight)**gamma_c,log_folder+gtp.seqname+'_{:04d}_tbd3do{}.png'.format(kk,tempi))


                if do_sota18:
                    imwrite(crop_only(est_hs_sota18[...,0],bbox_tight)**gamma_c,log_folder+gtp.seqname+'_{:04d}_sota18_{}.png'.format(kk,0))
                    imwrite(crop_only(est_hs_sota18[...,-1],bbox_tight)**gamma_c,log_folder+gtp.seqname+'_{:04d}_sota18_{}.png'.format(kk,7))

                if do_deblurgan:
                    # imwrite(crop_only(est_hs_deblurgan,bbox_tight)**gamma_c,log_folder+gtp.seqname+'_{:04d}_deblurgan.png'.format(kk))
                    imwrite(crop_only(est_hs_deblurgan_hr,bbox_tight)**gamma_c,log_folder+gtp.seqname+'_{:04d}_deblurgan_hr.png'.format(kk))
                   

            I[I<0]=0
            B[B<0]=0
            imwrite(crop_only(I,bbox_tight)**gamma_c,log_folder+gtp.seqname+'_{:04d}_im.png'.format(kk))
            imwrite(crop_only(B,bbox_tight)**gamma_c,log_folder+gtp.seqname+'_{:04d}_bgr.png'.format(kk))

