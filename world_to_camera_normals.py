import numpy as np
import os
import glob
import matplotlib.pyplot as plt

def world_to_cam_normals(folder):
    if folder != "" :
        if os.path.exists(folder) :
            if os.path.exists(folder+"cameras_unpacked.npz"):
                data_cam = np.load(folder+"cameras_unpacked.npz")
                R = data_cam["R"]
                imp = glob.glob(folder+"mesh_normal/*")
                for k,p in enumerate(imp) :
                    im = plt.imread(p)
                    s0,s1,s2 = im.shape
                    im_v = (im.reshape(-1,3).T)*2.0 -1.0
                    im_v_cam = (R[k,:,:] @ im_v)
                    im_v_cam[1,:] = -im_v_cam[1,:]
                    im_v_cam[2,:] = -im_v_cam[2,:]
                    im_v_cam = ((im_v_cam+1.0)/2.0).clip(0.0,1.0)
                    im_cam = (im_v_cam.T).reshape(s0,s1,s2)
                    plt.imsave(p,im_cam)

            else :
                raise("There is no cameras.npz in your folder")

        else :
            raise("Your folder doesn't exist !")
    else :
        raise("You need to add the folder : --folder name")
