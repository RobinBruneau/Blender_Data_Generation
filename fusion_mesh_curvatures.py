import glob

import numpy as np
import trimesh
import matplotlib.pyplot as plt


def modify_obj(file_path,file_path_save,colors):

    f2 = open(file_path_save,'w')
    ind = 0
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                data = line.split("\n")[0] + " {} {} {}\n".format(colors[ind,0],colors[ind,1],colors[ind,2])
                ind += 1
                f2.write(data)
            else :
                f2.write(line)
    f2.close()

folder = "D:/PhD/Dropbox/CVPR_2024_2/all_gt/"


cases = ["bear","buddha","cow","pot2","reading"]

for case in cases :

    with open(folder+case+"PNG_Mesh_GT.txt", "r") as file:
        lines = file.readlines()

    # Parse the lines to extract k1 and k2 values
    lk1 = []
    lk2 = []
    for line in lines:
        k1 = float(line.split("k1 =")[1].split(",")[0])
        k2 = float(line.split("k2 =")[1].split("\n")[0])
        lk1.append(k1)
        lk2.append(k2)

    lk1 = np.array(lk1)
    lk2 = np.array(lk2)
    both = np.concatenate((np.abs(np.copy(lk1)).reshape(-1,1),np.abs(np.copy(lk2)).reshape(-1,1)),axis=1)

    ind = np.argmax(both,axis=1)
    ind_1 = np.where(ind==1)

    curv = np.copy(lk1)
    curv[ind_1] = lk2[ind_1]
    curv = curv.clip(-2,2)
    min_curv = np.min(curv)
    max_curv = np.max(curv)
    curv = (curv-min_curv)/(max_curv-min_curv)

    curv_seg = np.ones_like(curv)*0.5
    curv_seg[np.where(curv<0.20)] = 0
    curv_seg[np.where(curv>0.80)] = 1

    #color_curv = plt.cm.jet(curv)[:, :3]
    #modify_obj(folder+case+"PNG_Mesh_GT.obj",folder + case + "PNG_Mesh_GT_CURV.obj",color_curv)

    color_curv = plt.cm.jet(curv_seg)[:, :3]
    modify_obj(folder + case + "PNG_Mesh_GT.obj",folder + case + "PNG_Mesh_GT_CURV_SEG_DUR.obj", color_curv)


a=0