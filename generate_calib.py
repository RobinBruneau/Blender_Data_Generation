import pickle
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

folder = 'D:\PhD\Projects\Playing_with_NeuS\data\MVPS_graphosoma_testing_raw\images_ps/'
data = np.load(folder+"lights.npz")
R  = np.load(folder+"R.npy")

'''
data_m = {}
dlc = []
for k,e in enumerate(data.files) :
    d = data[e]
    dc = R[:,:,k] @ d.transpose()
    dc = dc.transpose()
    dlc.append(dc)
data_m.update({"calib":dlc})
scipy.io.savemat(folder+"calib.mat",data_m)
'''
import os
#os.mkdir(folder+"normals/")
for k in range(20):

    name = (3-len(str(k)))*"0"+str(k)
    fold = folder + name + "/normal.jpg"
    data_ = (plt.imread(fold)/255.0)*2.0 -1.0
    mask = plt.imread(folder+name+"/mask.bmp")
    ind_0 = np.where(mask ==0)
    norm_data = np.linalg.norm(data_,axis=2)
    data = np.reshape(data_,(-1,3))
    data_world = (R[:,:,k].T @ data.T).T
    data_world = np.clip((data_world.reshape(1080,1920,3)+1.0)/2.0,0,1)
    data_world[ind_0] = 0
    plt.imsave(folder+"normals/"+name+".jpg",data_world)

    a=0