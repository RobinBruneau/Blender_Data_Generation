import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2

folder = "D:/PhD/Projects/Playing_with_NeuS/data/Graphosoma_RMVS_50_2/normals/"
imp = glob.glob(folder+"*")

for ip in imp :
    im = cv2.imread(ip)/255.0
    normals = im * 2.0 -1.0
    normals_norm = np.linalg.norm(normals,axis=2)
    mask = np.zeros((1080,1920,1))

    for k in range(1080):
        for j in range(1920):
            if normals_norm[k,j]>0.85 and normals_norm[k,j]<1.15:
                mask[k,j,:] = 255
                n = normals[k,j,:]
                nn = np.linalg.norm(n)
                n_normalized = n/nn
                n_color = (n_normalized+1.0)/2.0
                n_color_255 = n_color*255.0
                n_color_255_int = np.clip(np.round(n_color_255),0,255).astype(int)
                im[k,j,:] = n_color_255_int
            else :
                im[k,j,:] = 128
    mask = mask.astype(int)
    cv2.imwrite(ip,np.concatenate((im,mask),axis=2))