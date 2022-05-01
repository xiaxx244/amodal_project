import os
import cv2
import json
import numpy as np
import natsort
import glob
MASK_WIDTH = 1920				    # Dimensions should match those of ground truth image
MASK_HEIGHT = 1208
files = [ f.path for f in os.scandir("/media/bizon/Elements/OS2_roadlabel/") if f.is_dir() ]
im_path=natsort.natsorted(glob.glob('/media/bizon/Elements/amodal_dataset3/baseline_os2/real_A/*.png'),reverse=False)
def findlabel(im1,im2):
    for i in range(len(im1)):
        for j in range(len(im1[i])):
            if im1[i,j]!=0:
                im2[i,j]=1
    return im2
for file in files:
    mask_path=natsort.natsorted(glob.glob(file+'/*.npy'),reverse=False)
    for i in range(len(mask_path)):
        info=np.load(mask_path[i])
        mask = np.zeros((MASK_HEIGHT, MASK_WIDTH,3),dtype=np.uint8)
        print(mask_path[i])
        mask=findlabel(info,mask)
        #print(indices_list)
        #mask[indices_list]=1
        mask=mask[0:1200,0:1920]
        mask=cv2.resize(mask,(512,320))
        cv2.imwrite("/media/bizon/Elements/OS2_roadlabel/mask/"+im_path[i].split("/")[-1],mask)
    break
