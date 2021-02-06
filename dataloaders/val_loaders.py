import torch
from torchvision.utils import save_image
from dataloaders.loader import *
import os

def get_val_train_batch(device, log_path=None):
    val_batch, val_gt_paths = get_training_sample(["can"],min_obj=5,max_obj=5)
    val_batch2, _ = get_training_sample(["bowl"],min_obj=5,max_obj=5)
    val_batch2 = val_batch2.to(device).unsqueeze(0)
    val_batch = val_batch.to(device).unsqueeze(0)
    val_batch = torch.cat((val_batch,val_batch2),0)
    val_ind = 0
    val_gt_batch = get_gt_sample(val_gt_paths, val_ind).to(device).unsqueeze(0).unsqueeze(0)
    if log_path is not None:
        save_image((val_batch[0,:3,:,:].clone()+2)/4, os.path.join(log_path,'epoch{:04d}_im.png'.format(0)))
        save_image((val_batch[1,:3,:,:].clone()+2)/4, os.path.join(log_path,'epoch{:04d}_im2.png'.format(0)))
        save_image(val_gt_batch[0,0,:,:,:].clone(), os.path.join(log_path,'epoch{:04d}_gt.png'.format(0)))
    return val_batch, val_gt_batch

def get_vis_batch(log_path=None):
    val_batch, val_gt_paths = get_training_sample(["can"],min_obj=5,max_obj=5)
    val_batch = val_batch.unsqueeze(0)
    val_ind = 0
    if log_path is not None:
        save_image((val_batch[0,:3,:,:].clone()+2)/4, os.path.join(log_path,'epoch{:04d}_im.png'.format(0)))
        val_gt_batch = get_gt_sample(val_gt_paths, val_ind)
        save_image(val_gt_batch, os.path.join(log_path,'epoch{:04d}_gt.png'.format(0)))
    return val_batch

def eval_vis_batch(encoder, rendering, device, val_batch, log_path, epoch):
    latent = encoder(val_batch)
    t_tensor = torch.FloatTensor([0]).to(device).repeat(latent.shape[0], 1, latent.shape[2], latent.shape[3])
    t2_tensor = torch.FloatTensor([1]).to(device).repeat(latent.shape[0], 1, latent.shape[2], latent.shape[3])
    result = rendering(torch.cat((t_tensor,latent),1)).unsqueeze(1)
    result2 = rendering(torch.cat((t2_tensor,latent),1)).unsqueeze(1)
    save_image(result[0], os.path.join(log_path,'epoch{:04d}_0.png'.format(epoch+1)))
    save_image(result2[0], os.path.join(log_path,'epoch{:04d}_1.png'.format(epoch+1)))
    # val_loss1 = fmo_loss(result[:1], val_gt_batch)
    # val_loss2 = fmo_loss(result2[:1], val_gt_batch)

