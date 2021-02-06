import os
import argparse
import torch
import numpy as np
from torchvision.utils import save_image

from main_settings import g_resolution_x, g_resolution_y
from models.encoder import EncoderCNN
from models.rendering import RenderingCNN
from dataloaders.loader import get_transform
import cv2

from utils import *
import pdb

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--im", default=None, required=False)
	parser.add_argument("--bgr", default=None, required=False)
	parser.add_argument("--video", default=None, required=False)
	parser.add_argument("--steps", default=24, required=False)
	parser.add_argument("--models", default='saved_models', required=False)
	parser.add_argument("--output", default='output', required=False)
	return parser.parse_args()

def main():
	args = parse_args()

	if not os.path.exists(args.output):
		os.makedirs(args.output)

	device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
	torch.backends.cudnn.benchmark = True
	encoder = EncoderCNN()
	rendering = RenderingCNN()

	if torch.cuda.is_available():
		encoder.load_state_dict(torch.load(os.path.join(args.models, 'encoder_best.pt')))
		rendering.load_state_dict(torch.load(os.path.join(args.models, 'rendering_best.pt')))
	else:
		encoder.load_state_dict(torch.load(os.path.join(args.models, 'encoder_best.pt'),map_location=torch.device('cpu')))
		rendering.load_state_dict(torch.load(os.path.join(args.models, 'rendering_best.pt'),map_location=torch.device('cpu')))
		
	encoder = encoder.to(device)
	rendering = rendering.to(device)

	encoder.train(False)
	rendering.train(False)
	preprocess = get_transform()

	if not args.im is None and not args.bgr is None:
		I = imread(args.im)
		B = imread(args.bgr)
		bbox, radius = fmo_detect_maxarea(I,B)
		bbox = extend_bbox(bbox.copy(),4*np.max(radius),g_resolution_y/g_resolution_x,I.shape)
		# bbox = np.array([0,0,I.shape[0],I.shape[1]])
		im_crop = crop_resize(I, bbox, (g_resolution_x, g_resolution_y))
		bgr_crop = crop_resize(B, bbox, (g_resolution_x, g_resolution_y))
		input_batch = torch.cat((preprocess(im_crop), preprocess(bgr_crop)), 0).to(device).unsqueeze(0).float()
		with torch.no_grad():
			latent = encoder(input_batch)
			times = torch.linspace(0,1,args.steps).to(device)
			renders = rendering(latent,times[None])
		renders_rgba = renders[0].data.cpu().detach().numpy().transpose(2,3,1,0)
		## generate results
		tsr = rgba2hs(renders_rgba, bgr_crop)
		tsr_original = rev_crop_resize(tsr,bbox,B.copy())
		out = cv2.VideoWriter(os.path.join(args.output, 'tsr.avi'),cv2.VideoWriter_fourcc(*"MJPG"), 6, (I.shape[1], I.shape[0]),True)
		for ki in range(args.steps):
			imwrite(tsr_original[...,ki],os.path.join(args.output,'tsr{}.png'.format(ki)))
			out.write( (tsr_original[:,:,[2,1,0],ki] * 255).astype(np.uint8) )
		out.release()
	elif not args.video is None:
		pdb.set_trace()
	else:
		print('You should either provide both --im and --bgr, or --video.')

	

if __name__ == "__main__":
    main()