import os.path
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

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


def cameras_npz_unpacked(folder="",camera_file=""):
    if folder != "" :
        if os.path.exists(folder) :
            if os.path.exists(camera_file):
                data_cam = np.load(camera_file)
                nb_views = len(data_cam.files)//2
                lk = []
                lr = []
                lt = []
                lr_euler = []
                for k in range(nb_views):
                    P_k = data_cam["world_mat_{}".format(k)]
                    K,RT = load_K_Rt_from_P(P_k[:3,:])
                    lr.append(RT[:3,:3].T)
                    lt.append(-RT[:3,:3].T @ RT[:3,[3]])
                    lk.append(K[:3,:3])

                    rb = np.eye(3)
                    rb[1,1] = -1
                    rb[2,2] = -1
                    r = Rotation.from_matrix((rb @ RT[:3,:3].T).T)
                    euler_rot = r.as_euler('xyz', degrees=True)
                    lr_euler.append(euler_rot)

                data_out = {"K":lk,"R":lr,"T":lt,"R_euler":lr_euler}
                np.savez(folder+"cameras_unpacked.npz",**data_out)
            else :
                raise("There is no cameras.npz in your folder")

        else :
            raise("Your folder doesn't exist !")
    else :
        raise("You need to add the folder : --folder name")
