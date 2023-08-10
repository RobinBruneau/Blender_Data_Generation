import numpy as np
import cv2
import os
import glob
import shutil

folder = "/home/robin/Desktop/PhD/Projects/RPS_fonctionne/Data_salle/Data_ortho_normal_full/"

interface = np.load(folder+"interface.npz")
vertices = interface["vertices"]
faces = interface["faces"]
normals = interface["normals"]
RR = np.load(folder + "R.npy")
TT = np.load(folder + "T.npy")
K = np.load(folder + "K.npy")

a=0

if not os.path.exists(folder+"cmp_normal_ps/"):
    os.mkdir(folder + "cmp_normal_ps/")

for k in range(RR.shape[2]):

    if not os.path.exists(folder + "cmp_normal_ps/"+(3-len(str(k)))*"0"+str(k)+"/"):
        os.mkdir(folder + "cmp_normal_ps/"+(3-len(str(k)))*"0"+str(k)+"/")

    mask = cv2.imread(folder+"mask/"+(3-len(str(k)))*"0"+str(k)+".png")
    ind_remove = np.where(mask < 10)
    im = 255.0*np.ones((1080,1920,3))

    R = RR[:,:,k].reshape(3,3)
    T = TT[:,[k]]

    v_cam = R @ vertices + T
    v_cam[2,:] = 1.0
    v_pixels = K @ v_cam

    dir_cam = (R.T @ np.array([0,0,1]).reshape(3,1)).reshape(3)

    for p_id in range(faces.shape[1]):
        if np.dot(-dir_cam,normals[:,p_id]) > 1e-8 :
            face = faces[:,p_id].astype(int)
            triangle_cnt = np.array([v_pixels[:2, face[0]], v_pixels[:2, face[1]], v_pixels[:2, face[2]]])
            triangle_cnt = triangle_cnt.reshape((-1, 1, 2)).astype(np.int32)
            coco = (p_id, p_id, p_id)
            cv2.drawContours(im, [triangle_cnt], 0, coco, -1)

    im[ind_remove] = 255.0
    cv2.imwrite(folder+"cmp_normal_ps/"+(3-len(str(k)))*"0"+str(k)+"/mask.bmp",im)

    all_im = glob.glob(folder+"image/"+(3-len(str(k)))*"0"+str(k)+"_*.png")

    for i in range(len(all_im)):
        name = all_im[i].split((3-len(str(k)))*"0"+str(k)+"_")[-1]
        shutil.copy(all_im[i],folder+"cmp_normal_ps/"+(3-len(str(k)))*"0"+str(k)+"/"+name)




