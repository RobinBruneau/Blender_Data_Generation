import os.path

import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2

folder = "/home/robin/Desktop/PhD/Data/Data_ortho_normal_sphere_area/"
RR = np.load(folder + "R.npy")
TT = np.load(folder + "T.npy")
K = np.load(folder + "K.npy")

# images cube
im_cube = []
im_mask = []
for k in range(RR.shape[2]):
    im_cube.append(np.sum(cv2.imread(folder+"mask_medium/"+"0"*(3-len(str(k)))+str(k)+".png"),axis=2))
    im_mask.append(np.sum(cv2.imread(folder+"mask/"+"0"*(3-len(str(k)))+str(k)+".png"),axis=2))
im_plan = [255*np.ones(i.shape) for i in im_cube]

interface = np.load(folder + "interface.npz")

for k in range(interface["faces"].shape[1]):
    ff = interface["faces"][:,k].astype(int)
    vv = interface["vertices"][:,ff]
    nn = interface["normals"][:, k]
    nn = nn / np.linalg.norm(nn)
    for ll,im in enumerate(im_cube) :
        R = RR[:,:,ll].reshape(3,3)
        T = TT[:,[ll]]
        cam_direction = (R.T @ (np.array([0, 0, 1]).reshape(3, 1))).reshape(3)
        if np.dot(nn, -cam_direction) > 1e-8:
            pts_cam =R @ vv + T
            pts_cam[2,:] = 1.0
            pts_pixels = K @ pts_cam
            triangle_cnt = np.array([pts_pixels[:2,0], pts_pixels[:2,1], pts_pixels[:2,2], pts_pixels[:2,0]])
            triangle_cnt = triangle_cnt.reshape((-1, 1, 2)).astype(np.int32)
            coco = (k,k,k)
            cv2.drawContours(im_plan[ll], [triangle_cnt], 0, coco, -1)


if not os.path.exists(folder+"plan_id/"):
    os.mkdir(folder+"plan_id/")

for k,plan in enumerate(im_plan):
    cv2.imwrite(folder+"plan_id/"+"0"*(3-len(str(k)))+str(k)+".png",plan)



