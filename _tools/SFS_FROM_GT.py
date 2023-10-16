import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import trimesh
import pyoctree as ot

folder = "D:\PhD\Dropbox\Data\Data\Data_ortho_normal_sphere/"
RR = np.load(folder + "R.npy")
TT = np.load(folder + "T.npy")
K = np.load(folder + "K.npy")
interface = np.load(folder + "interface.npz")
vertices = interface["vertices"]
faces = interface["faces"].astype(int)
normals = interface["normals"]
normals = normals / np.linalg.norm(normals,axis=0)
d = np.zeros(faces.shape[1])
v_min = np.min(vertices,axis=1)
v_max = np.max(vertices,axis=1)
n = 300j
X,Y,Z = np.mgrid[v_min[0]:v_max[0]:n, v_min[1]:v_max[1]:n, v_min[2]:v_max[2]:n]
XYZ = np.stack((X,Y,Z))
X = X.reshape(1,-1)
Y = Y.reshape(1,-1)
Z = Z.reshape(1,-1)
voxels = np.concatenate((X,Y,Z),axis=0)
del X,Y,Z



voxels_max = voxels.shape[1]
probability = (np.linalg.norm(voxels,axis=0) <= 1.07/2).astype(float)
probability = probability.reshape(int(np.imag(n)),int(np.imag(n)),int(np.imag(n)))
data_grid = {"grid": probability,"XYZ":XYZ}
np.savez(folder+"grid.npz",**data_grid)

import scipy.io
data_grid = {"grid": probability}
scipy.io.savemat(folder + 'grid.mat', data_grid)

