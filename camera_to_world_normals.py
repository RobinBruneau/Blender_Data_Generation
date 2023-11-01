import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import cv2
def load_K_Rt_from_P(P):

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

folder = "D:/PhD/Dropbox/CVPR_2024/check_normals_and_visibility/readingPNG/"
camera_data = np.load(folder+"cameras.npz")

all_R = []
for k in range(len(camera_data.files)//2):
    P = camera_data["world_mat_{}".format(k)]
    _,RT = load_K_Rt_from_P(P[:3,:])
    R = RT[:3,:3]
    all_R.append(R)

imp = glob.glob(folder+"input_SDM/normal/*")
for k,p in enumerate(imp) :
    im = plt.imread(p)
    s0,s1,s2 = im.shape
    im_v_cam = (im.reshape(-1,3).T)*2.0 -1.0
    im_v_cam[1, :] = -im_v_cam[1, :]
    im_v_cam[2, :] = -im_v_cam[2, :]
    im_v = (all_R[k] @ im_v_cam)
    im_v = ((im_v+1.0)/2.0).clip(0.0,1.0)
    im_cam = (im_v.T).reshape(s0,s1,s2)
    plt.imsave(p,im_cam)

