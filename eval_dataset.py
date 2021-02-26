import numpy as np
import pdb
import os
import torch
from torchvision.utils import save_image

from vis import *
from main_settings import *
from shutil import copyfile

from models.encoder import *
from models.rendering import *
from models.loss import *
from dataloaders.loader import *
from dataloaders.tbd_loader import *
from helpers.torch_helpers import *
from renderer.run_addbg import *
from helpers.paper_helpers import *

g_saved_models_folder = './saved_models/'

def main():
	print(torch.__version__)

	gpu_id = 0
	train_mode = False

	device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")
	print(device)
	torch.backends.cudnn.benchmark = True
	encoder = EncoderCNN()
	rendering = RenderingCNN()

	if torch.cuda.is_available():
		encoder.load_state_dict(torch.load(os.path.join(g_saved_models_folder, 'encoder_best.pt')))
		rendering.load_state_dict(torch.load(os.path.join(g_saved_models_folder, 'rendering_best.pt')))
	else:
		encoder.load_state_dict(torch.load(os.path.join(g_saved_models_folder, 'encoder_best.pt'),map_location=torch.device('cpu')))
		rendering.load_state_dict(torch.load(os.path.join(g_saved_models_folder, 'rendering_best.pt'),map_location=torch.device('cpu')))
		
	encoder = encoder.to(device)
	rendering = rendering.to(device)

	encoder.train(train_mode)
	rendering.train(train_mode)

	encoder_params = sum(p.numel() for p in encoder.parameters())
	rendering_params = sum(p.numel() for p in rendering.parameters())
	print('Encoder params {:2f}M, rendering params {:2f}M'.format(encoder_params/1e6,rendering_params/1e6))

	## full evaluation on datasets
	if False:
		datasets = ['tbd','tbd3d','tbdfalling','wildfmo','youtube']
		evaluate_on(encoder, rendering, device, datasets[2])

	## run on falling objects 
	if True:
		get_figure_images(encoder, rendering, device, 'tbdfalling', 2, 31+2, results_mode=True, n_occ=7)
		# get_figure_images(encoder, rendering, device, 'tbdfalling', 0, 4, results_mode=True)
		# get_figure_images(encoder, rendering, device, 'tbdfalling', 1, 7, results_mode=True)
		# get_figure_images(encoder, rendering, device, 'tbdfalling', 2, 2, results_mode=True)
		# get_figure_images(encoder, rendering, device, 'tbdfalling', 3, 2, results_mode=True)
		# get_figure_images(encoder, rendering, device, 'tbdfalling', 4, 7, results_mode=True)
		# get_figure_images(encoder, rendering, device, 'tbdfalling', 5, 6, results_mode=True)

	## run on tbd-3d
	if False:
		get_figure_images(encoder, rendering, device, 'tbd3d', 1, 13, results_mode=True)
		get_figure_images(encoder, rendering, device, 'tbd3d', 2, 1, results_mode=True)
		get_figure_images(encoder, rendering, device, 'tbd3d', 6, 23, results_mode=True)
		get_figure_images(encoder, rendering, device, 'tbd3d', 7, 19, results_mode=True)

	## run on tbd
	if False:
		get_figure_images(encoder, rendering, device, 'tbd', 0, 1, results_mode=True, n_occ=7)
		get_figure_images(encoder, rendering, device, 'tbd', 'volleyball', 43, results_mode=True, n_occ=7)

	# run on synthetic data
	if False:
		objid = np.nonzero([temp == 'mug' for temp in g_render_objs_train])[0][0]
		get_figure_images(encoder, rendering, device, 'train', objid, 123, results_mode=True, n_occ=7)


if __name__ == "__main__":
    main()