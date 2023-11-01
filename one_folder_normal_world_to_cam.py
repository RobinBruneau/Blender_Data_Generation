import subprocess
import os
import argparse
from camera_npz_to_npy import *
from world_to_camera_normals import *

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


folder = "D:/PhD/Projects/Playing_with_NeuS/data/Boudha_marble_1920_1080_MVPS/"
camera_file = folder + "cameras.npz"

data_cam = np.load(camera_file)
nb_views = len(data_cam.files)
lk = []
lr = []
lt = []
for k in range(nb_views):
    P_k = data_cam["world_mat_{}".format(k)]
    K,RT = load_K_Rt_from_P(P_k[:3,:])
    lr.append(RT[:3,:3].T)
    lt.append(-RT[:3,:3].T @ RT[:3,[3]])
    lk.append(K[:3,:3])



R = lr
imp = glob.glob(folder+"normal/*")
for k,p in enumerate(imp) :
    im = plt.imread(p)
    s0,s1,s2 = im.shape
    im_v = (im.reshape(-1,3).T)*2.0 -1.0
    im_v_cam = (R[k] @ im_v)
    im_v_cam[1,:] = -im_v_cam[1,:]
    im_v_cam[2,:] = -im_v_cam[2,:]
    im_v_cam = ((im_v_cam+1.0)/2.0).clip(0.0,1.0)
    im_cam = (im_v_cam.T).reshape(s0,s1,s2)
    plt.imsave(p,im_cam)
