import torch.nn as nn
import torch
from helpers.torch_helpers import *
from main_settings import *

class FMOLoss(nn.Module):
    def __init__(self, reduction='none'):
        super(FMOLoss, self).__init__()
        self.reduction = reduction

    def forward(self, renders_all, hs_frames, input_batch, latents=None):
        assert renders_all.shape[0] == hs_frames.shape[0], "predict & target batch size don't match"
        renders = renders_all[:,:,:4]
        renders_supervised = renders[:,:g_fmo_train_steps]
        renders_supervised_all = renders_all[:,:g_fmo_train_steps]

        supervised_loss = 0
        if g_use_supervised:
            loss1 = fmo_loss(renders_supervised, hs_frames)
            loss2 = fmo_loss(renders_supervised, torch.flip(hs_frames,[1]))
            supervised_loss,_ = torch.cat((loss1.unsqueeze(0),loss2.unsqueeze(0)),0).min(0)

        loss_timecons = 0*supervised_loss
        if g_use_selfsupervised_timeconsistency:
            if g_timeconsistency_type == 'oflow':
                loss_timecons = 10*oflow_loss(renders_supervised_all)
            else:
                loss_timecons = 5*temporal_consistency_loss(renders_supervised)

        if not g_use_supervised:
            supervised_loss = 0*loss_timecons

        model_loss = 0*supervised_loss
        if g_use_selfsupervised_model:
            modelled_renders = torch.cat( (renders[:,:,:3]*renders[:,:,3:], renders[:,:,3:]), 2).mean(1)

            region_of_interest = None
            if g_use_supervised:
                region_of_interest = hs_frames[:,:,3:].mean(1) > 0
            
            model_loss = fmo_model_loss(input_batch, modelled_renders, Mask = region_of_interest)
        
        loss_sharp_mask = 0*supervised_loss
        if g_use_selfsupervised_sharp_mask:
            if g_sharp_mask_type == 'entropy':
                loss_sharp_mask = mask_sharp_loss_entropy_batchsum(renders) ## / 4 for similar scale as x*(1-x)
            else:
                loss_sharp_mask = mask_sharp_loss_batchsum(renders)
            loss_sharp_mask /= g_fmo_steps

        loss_latent = 0*supervised_loss
        if g_use_latent_learning:
            # normalization = nn.MSELoss(reduction='none')(latents[0],torch.flip(latents[1],[0])).mean([1,2,3]).max()
            loss_latent = nn.MSELoss(reduction='none')(latents[0],latents[1]).mean([1,2,3]) # / normalization

        loss = supervised_loss + loss_timecons + model_loss + loss_sharp_mask + loss_latent

        if self.reduction == 'none':
            return supervised_loss, model_loss, loss_sharp_mask, loss_timecons, loss_latent, loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

def oflow_loss(renders):
    time_inc = 1/(g_fmo_steps-1)
    imgs1 = renders[:,:-1,:4]
    flows = time_inc*renders[:,:-1,4:]
    imgs2 = renders[:,1:,:4]
    new_shape_imgs = (flows.shape[0]*flows.shape[1],imgs1.shape[2],imgs1.shape[3],imgs1.shape[4])
    new_shape_flows = (flows.shape[0]*flows.shape[1],flows.shape[2],imgs1.shape[3],imgs1.shape[4])
    img1_wrapped = flow_warp(torch.reshape(imgs1,new_shape_imgs), torch.reshape(flows,new_shape_flows))
    loss_o = nn.L1Loss(reduction='none')(imgs2, torch.reshape(img1_wrapped, imgs1.shape))*renders[:,1:,3:4]
    return loss_o.mean([1,2,3,4])

def flow_warp(img, flow):
    """Warp an image or feature map with optical flow (from https://www.programcreek.com/python/example/104458/torch.nn.functional.grid_sample)
    Args:
        x (Tensor): size (N, C, H, W)
        flow (Tensor): size (N, H, W, 2), normal value
        interp_mode (str): 'nearest' or 'bilinear'
        padding_mode (str): 'zeros' or 'border' or 'reflection'

    Returns:
        Tensor: warped image or feature map
    """
    B, C, H, W = img.size()
    # mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False
    grid = grid.type_as(img)
    vgrid = grid[None] + flow.permute(0,2,3,1)
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = torch.nn.functional.grid_sample(img, vgrid_scaled, mode='bilinear', padding_mode='border')
    return output 

def fmo_loss(Yp, Y):
    YM = Y[:,:,-1:,:,:]
    YpM = Yp[:,:,-1:,:,:]
    YF = Y[:,:,:3]
    YpF = Yp[:,:,:3]
    YMb = (YM > 0).type(YpM.dtype)

    mloss = 0.5*batch_loss(YpM, YM, YMb) + 0.5*batch_loss(YpM, YM, (1-YMb))

    floss = batch_loss(YpF*YpM, YF*YM, YMb[:,:,[0,0,0]])
    
    loss = 2*mloss + floss
    return loss

def batch_loss(YpM, YM, YMb):
    losses = nn.L1Loss(reduction='none')(YpM*YMb, YM*YMb)
    if len(losses.shape) > 4:
        bloss = losses.sum([1,2,3,4]) / YMb.sum([1,2,3,4])
    else:
        bloss = losses.sum([1,2,3]) / (YMb.sum([1,2,3]) + 0.01)
    return bloss

def fmo_loss_v1(Yp, Y):
    YM = Y[:,:,-1:,:,:]
    YpM = Yp[:,:,-1:,:,:]
    YF = Y[:,:,:3]
    YpF = Yp[:,:,:3]
    YMb = YM > 0
    YMbnot = ~YMb

    mloss = 0.5*nn.L1Loss()(YpM[YMb], YM[YMb]) + 0.5*nn.L1Loss()(YpM[YMbnot], YM[YMbnot])
    
    YMb3 = YMb[:,:,[0,0,0]]
    # floss = nn.L1Loss()( (YpF)[YMb3], (YF)[YMb3])
    floss = nn.L1Loss()( (YpF*YpM)[YMb3], (YF*YM)[YMb3])
    loss = mloss + 2*floss
    return loss
    
def fmo_model_loss(input_batch, renders, Mask = None):
    expected = input_batch[:,3:] * (1 - renders[:,3:]) + renders[:,:3]
    if Mask is None:
        Mask = renders[:,3:] > 0.05
    Mask = Mask.type(renders.dtype)
    model_loss = batch_loss(expected, input_batch[:,:3], Mask)
    # losses = nn.L1Loss(reduction='none')(expected, input_batch[:,:3])
    # model_loss = losses.mean([1,2,3])
    return model_loss

def mask_sharp_loss(renders):
    loss = renders[:,3] * (1 - renders[:,3])
    return torch.mean(loss, [1,2])

def mask_sharp_loss_batchsum(renders):
    loss = renders[:,:,3] * (1 - renders[:,:,3])
    return torch.mean(loss, [2,3]).sum(1)

def mask_sharp_loss_entropy_batchsum(renders, eps=1e-12):
    posr = renders[:,:,3]
    logp = torch.log2(posr+eps)
    negr = 1-posr
    logn = torch.log2(negr+eps)
    loss = -posr*logp - negr*logn
    return torch.mean(loss, [2,3]).sum(1)

def temporal_consistency_loss(renders):
    # nccs = 1-normalized_cross_correlation(renders[:,:-1,-1], renders[:,1:,-1])
    nccs = 1-normalized_cross_correlation_channels(renders[:,:-1], renders[:,1:])
    return nccs.mean(1)