import pickle
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import shutil
import glob
from pathlib import Path
import cv2

folder = 'D:\PhD\Dropbox\Data\Data\DiLiGenT-MV\DiLiGenT-MV\MVPS//bearPNG/'
data = np.load(folder+"lights.npz")
R  = np.load(folder+"R.npy")
T = np.load(folder+"T.npy")
folder2 = "D:\PhD\Projects\Playing_with_NeuS\data\MVPS_graphosoma_normal_ps/"
R2  = np.load(folder2+"R.npy")
T2 = np.load(folder2+"T.npy")
'''
files = data.files
for e in files :
    if "pos" in e :
        files.remove(e)

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
'''
imp = glob.glob(folder+"normal_lissage/*")
for ip in imp :
    im = cv2.imread(ip)
    im_l= cv2.medianBlur(im,3)
    cv2.imwrite(ip,im_l)
'''

for k,e in enumerate(glob.glob("D:\PhD\Dropbox\Data\Data\DiLiGenT-MV\DiLiGenT-MV\MVPS//bearPNG//normal_ps/*.png")):

    name = e.split("\\")[-1].split(".")[0]
    data_ = plt.imread(e)
    data_ = data_*2.0 -1.0
    mask = plt.imread("D:\PhD\Dropbox\Data\Data\DiLiGenT-MV\DiLiGenT-MV\MVPS//bearPNG\mask/"+name+".png")
    ind_0 = np.where(mask ==0)
    norm_data = np.linalg.norm(data_,axis=2)
    data = np.reshape(data_,(-1,3))
    data_world = (R[k,:,:] @ data.T).T
    data_world = np.clip((data_world.reshape(data_.shape[0],data_.shape[1],3)+1.0)/2.0,0,1)
    data_world[ind_0] = 0
    plt.imsave(e,data_world)



'''
for k in range(20):

    name = (3-len(str(k)))*"0"+str(k)
    fold = folder+"/image_ps/"+name+ "/normal.png"
    data_ = plt.imread(fold)
    data_ = data_*2.0 -1.0
    mask = plt.imread(folder+"/image_ps/"+name+"/mask.bmp")
    ind_0 = np.where(mask ==0)
    norm_data = np.linalg.norm(data_,axis=2)
    data = np.reshape(data_,(-1,3))
    data_world = (R[:,:,k].T @ data.T).T
    data_world = np.clip((data_world.reshape(1080,1920,3)+1.0)/2.0,0,1)
    data_world[ind_0] = 0
    plt.imsave(folder+"normal_ps/"+name+".png",data_world)

    a=0
'''
'''

Path(folder+"/image_ps").mkdir(exist_ok=True)
for k in range(20):

    name = (3-len(str(k)))*"0"+str(k)
    Path(folder+"/image_ps/"+name+"/").mkdir(exist_ok=True)
    files = glob.glob(folder+'/image/'+name+"_*")
    for i,f in enumerate(files) :
        name_2 = (3-len(str(i)))*"0"+str(i)
        shutil.copy(f,folder+"/image_ps/"+name+"/"+name_2+".png")
    mask_file = folder+"/mask/"+name+".png"
    shutil.copy(mask_file,folder+"/image_ps/"+name+"/mask.bmp")
'''