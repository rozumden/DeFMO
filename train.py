import argparse
import numpy as np
import random
import pdb
import os
import torch
import time

from utils import *
from vis import *
from main_settings import *

from models.encoder import *
from models.rendering import *
from models.loss import *
from dataloaders.loader import *
from dataloaders.val_loaders import *
from helpers.torch_helpers import *
from torch.utils.tensorboard import SummaryWriter

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    encoder = EncoderCNN()
    rendering = RenderingCNN()
    loss_function = FMOLoss()

    if g_finetune:
        g_load_temp_folder = '/home.stud/rozumden/tmp/PyTorch/20200918_2239_consfm2'
        encoder.load_state_dict(torch.load(os.path.join(g_load_temp_folder, 'encoder.pt')))
        rendering.load_state_dict(torch.load(os.path.join(g_load_temp_folder, 'rendering.pt')))

    encoder = nn.DataParallel(encoder).to(device)
    rendering = nn.DataParallel(rendering).to(device)
    loss_function = nn.DataParallel(loss_function).to(device)

    if not os.path.exists(g_temp_folder):
        os.makedirs(g_temp_folder)

    log_path = os.path.join(g_temp_folder,'training')
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    encoder_params = sum(p.numel() for p in encoder.parameters())
    rendering_params = sum(p.numel() for p in rendering.parameters())
    encoder_grad = sum(int(p.requires_grad) for p in encoder.parameters())
    encoder_p = sum(1 for p in encoder.parameters())
    print('Encoder params {:2f}M, rendering params {:2f}M'.format(encoder_params/1e6,rendering_params/1e6))
    
    training_set = ShapeBlurDataset(dataset_folder=g_dataset_folder, render_objs = g_render_objs, number_per_category=g_number_per_category,do_augment=True,use_latent_learning=g_use_latent_learning)
    training_generator = torch.utils.data.DataLoader(training_set, batch_size=g_batch_size,shuffle=True,num_workers=g_num_workers,drop_last=True)
    val_set = ShapeBlurDataset(dataset_folder=g_validation_folder, render_objs = g_render_objs_val, number_per_category=g_number_per_category_val,do_augment=True,use_latent_learning=False)
    val_generator = torch.utils.data.DataLoader(val_set, batch_size=g_batch_size,shuffle=True,num_workers=g_num_workers,drop_last=True)

    vis_train_batch, _ = get_training_sample(["can"],min_obj=5,max_obj=5,dataset_folder=g_dataset_folder)
    vis_train_batch = vis_train_batch.unsqueeze(0).to(device)
    vis_val_batch, _ = get_training_sample(["can"],min_obj=4,max_obj=4,dataset_folder=g_validation_folder)
    vis_val_batch = vis_val_batch.unsqueeze(0).to(device)

    all_parameters = list(encoder.parameters()) + list(rendering.parameters())
    optimizer = torch.optim.Adam(all_parameters, lr=g_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    writer = SummaryWriter(log_path)

    train_losses = []
    val_losses = []
    best_val_loss = 100.0
    for epoch in range(g_epochs):
        encoder.train()
        rendering.train()

        t0 = time.time()
        supervised_loss = []
        model_losses = []
        sharp_losses = []
        timecons_losses = []
        latent_losses = []
        joint_losses = []
        for it, (input_batch, times, hs_frames, times_left) in enumerate(training_generator):
            input_batch, times, hs_frames, times_left = input_batch.to(device), times.to(device), hs_frames.to(device), times_left.to(device)

            renders = []

            if g_use_latent_learning:
                latent = encoder(input_batch[:,:6])
                latent2 = encoder(input_batch[:,6:])
            else:
                latent = encoder(input_batch)
                latent2 = []
            
            renders = rendering(latent, torch.cat((times,times_left),1))
            
            sloss, mloss, shloss, tloss, lloss, jloss = loss_function(renders, hs_frames, input_batch[:,:6], (latent,latent2))

            supervised_loss.append(sloss.mean().item())
            model_losses.append(mloss.mean().item())
            sharp_losses.append(shloss.mean().item())
            timecons_losses.append(tloss.mean().item())
            latent_losses.append(lloss.mean().item())

            jloss = jloss.mean()
            joint_losses.append(jloss.item())    
            if it % 50 == 0:
                print("Epoch {:4d}, it {:4d}".format(epoch+1, it), end =" ")
                if g_use_supervised:
                    print(", loss {:.3f}".format(np.mean(supervised_loss)), end =" ")
                if g_use_selfsupervised_model:
                    print(", model {:.3f}".format(np.mean(model_losses)), end =" ")
                if g_use_selfsupervised_sharp_mask:
                    print(", sharp {:.3f}".format(np.mean(sharp_losses)), end =" ")
                if g_use_selfsupervised_timeconsistency:
                    print(", time {:.3f}".format(np.mean(timecons_losses)), end =" ")
                if g_use_latent_learning:
                    print(", latent {:.3f}".format(np.mean(latent_losses)), end =" ")

                print(", joint {:.3f}".format(np.mean(joint_losses)))
            
            optimizer.zero_grad()
            jloss.backward()
            optimizer.step()
        train_losses.append(np.mean(supervised_loss))

        with torch.no_grad():
            encoder.eval()
            rendering.eval()
            
            running_losses_min = []
            running_losses_max = []
            for it, (input_batch, times, hs_frames, _) in enumerate(val_generator):
                input_batch, times, hs_frames = input_batch.to(device), times.to(device), hs_frames.to(device)
                latent = encoder(input_batch)
                renders = rendering(latent, times)[:,:,:4]

                val_loss1 = fmo_loss(renders, hs_frames)
                val_loss2 = fmo_loss(renders, torch.flip(hs_frames,[1]))
                losses = torch.cat((val_loss1.unsqueeze(0),val_loss2.unsqueeze(0)),0)
                min_loss,_ = losses.min(0)
                max_loss,_ = losses.max(0)
                running_losses_min.append(min_loss.mean().item())
                running_losses_max.append(max_loss.mean().item())
            print("Epoch {:4d}, val it {:4d}, loss {}".format(epoch+1, it, np.mean(running_losses_min)))
            val_losses.append(np.mean(running_losses_min))
            if val_losses[-1] < best_val_loss and epoch >= 0:
                torch.save(encoder.module.state_dict(), os.path.join(g_temp_folder, 'encoder_best.pt'))
                torch.save(rendering.module.state_dict(), os.path.join(g_temp_folder, 'rendering_best.pt'))
                best_val_loss = val_losses[-1]
                print('    Saving best validation loss model!  ')
            
            writer.add_scalar('Loss/train_supervised', train_losses[-1], epoch+1)
            writer.add_scalar('Loss/train_joint', np.mean(joint_losses), epoch+1)
            if g_use_selfsupervised_model:
                writer.add_scalar('Loss/train_selfsupervised_model', np.mean(model_losses), epoch+1)
            if g_use_selfsupervised_sharp_mask:
                writer.add_scalar('Loss/train_selfsupervised_sharpness', np.mean(sharp_losses), epoch+1)
            if g_use_selfsupervised_timeconsistency:
                writer.add_scalar('Loss/train_selfsupervised_timeconsistency', np.mean(timecons_losses), epoch+1)
            if g_use_latent_learning:
                writer.add_scalar('Loss/train_selfsupervised_latent', np.mean(latent_losses), epoch+1)
            writer.add_scalar('Loss/val_min', val_losses[-1], epoch+1)
            writer.add_scalar('Loss/val_max', np.mean(running_losses_max), epoch+1)
            writer.add_scalar('LR/value', optimizer.param_groups[0]['lr'], epoch+1)
            writer.add_images('Vis Train Batch', get_images(encoder, rendering, device, vis_train_batch)[0], global_step=epoch+1)
            writer.add_images('Vis Val Batch', get_images(encoder, rendering, device, vis_val_batch)[0], global_step=epoch+1)
            
            concat = torch.cat((renders[:,0],renders[:,-1],hs_frames[:,0],hs_frames[:,-1]),2)
            writer.add_images('Val Batch', concat[:,3:]*(concat[:,:3]-1)+1, global_step=epoch+1)
            
        time_elapsed = (time.time() - t0)/60
        print('Epoch {:4d} took {:.2f} minutes, lr = {}, av train loss {:.5f}, val loss min {:.5f} max {:.5f}'.format(epoch+1, time_elapsed, optimizer.param_groups[0]['lr'], train_losses[-1], val_losses[-1], np.mean(running_losses_max)))
        scheduler.step()
        
    # pdb.set_trace()
    torch.cuda.empty_cache()
    torch.save(encoder.module.state_dict(), os.path.join(g_temp_folder, 'encoder.pt'))
    torch.save(rendering.module.state_dict(), os.path.join(g_temp_folder, 'rendering.pt'))
    writer.close()

if __name__ == "__main__":
    main()