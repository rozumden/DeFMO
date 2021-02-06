import cv2
import numpy as np
import PIL
from PIL import Image

def get_visim(H,F,M,I=None):
	if I is not None:
		sz = I.shape[0]
	else:
		sz = H.shape[0]
	FM = np.r_[np.repeat(M[:,:,np.newaxis], 3, axis=2),F]
	FM = cv2.resize(FM,(0,0), fx=sz/FM.shape[0], fy=sz/FM.shape[0])
	visim = FM
	if H is not None:
		H0 = H
		if len(H.shape) == 2 or H.shape[2] == 1:
			H0 = np.repeat(H[:,:,np.newaxis], 3, axis=2)
		Hvis = H0/np.max(H0)
		visim = np.concatenate((Hvis, visim),1)
	if I is not None:
		visim = np.concatenate((I, visim),1)
	return visim

def imshow_nodestroy(im, inter=1):
	if inter == 1:
		cv2.imshow('image',im), cv2.waitKey(1)
	else:
		cv2.imshow('image',cv2.resize(im, (0,0), fx=inter, fy=inter, interpolation = cv2.INTER_NEAREST)), cv2.waitKey(1)


def imshow(im, wkey=0, inter=1):
	if type(im) == np.ndarray:
		im = im
	elif type(im) == PIL.PngImagePlugin.PngImageFile:
		im = np.array(im)
		if im.shape[2] > 1:
			im = im[:,:,[2,1,0]]
	elif type(im) != np.ndarray:
		im = im.data.cpu().detach().numpy()
		if len(im.shape) == 4:
			im = im[0,:,:,:]
		im = im.transpose(1,2,0)

	if inter == 1:
		cv2.imshow('image',im), cv2.waitKey(int(np.round(wkey*1000))), cv2.destroyAllWindows() 
	else:
		cv2.imshow('image',cv2.resize(im, (0,0), fx=inter, fy=inter, interpolation = cv2.INTER_NEAREST)), cv2.waitKey(int(np.round(wkey*1000))), cv2.destroyAllWindows()

def imshow6(im, wkey=0):
	imshow(im, wkey=wkey, inter=6)