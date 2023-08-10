import os
import concurrent.futures
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import scipy.io

def compute_refracted_ray_vecto(incident_rays, interface_normals, IOR_air,IOR_medium):
    incident_rays = incident_rays / np.linalg.norm(incident_rays,axis=0)
    interface_normals = interface_normals / np.linalg.norm(interface_normals,axis=0)

    eps = 1e-10

    refracted_rays = np.zeros(incident_rays.shape)

    ind_rays_para = np.sum(abs(np.cross(incident_rays,interface_normals,axis=0)),axis=0) < eps
    ind_refracted = np.logical_not(ind_rays_para)
    refracted_rays[:,ind_rays_para] = incident_rays[:,ind_rays_para]
    interface_normals[:,ind_rays_para] = 0

    cross = np.cross(incident_rays,interface_normals,axis=0)
    dot = np.sum(incident_rays*interface_normals,axis=0)
    a = np.linalg.norm(cross,axis=0)
    b = np.sign(dot)
    c = np.tan(np.arcsin(a* IOR_air / IOR_medium))
    coeff = a*b/c - dot
    refracted_rays[:,ind_refracted] = incident_rays[:,ind_refracted] + coeff[ind_refracted] * interface_normals[:,ind_refracted]

    return refracted_rays


folder = "/home/robin/Desktop/PhD/Data/Data_ortho_normal_sphere_area/"
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

all_mask = []
for nb_view in range(RR.shape[2]):
    all_mask.append(np.sum(cv2.imread(folder+"mask/"+"0"*(3-len(str(nb_view)))+str(nb_view)+".png"),axis=2)>10)
all_mask = np.array(all_mask)

for face_id in range(normals.shape[1]):

    vert_0 = vertices[:,faces[0,face_id]]
    normal_id = normals[:,face_id]
    d[face_id] = -np.dot(vert_0,normal_id)


def compute_SFS(vox,k):

    count_positive = np.zeros(vox.shape[1])
    count_negative = np.zeros(vox.shape[1])
    ind = np.linspace(0,vox.shape[1]-1,vox.shape[1]).astype(int)

    for nb_view in tqdm(range(RR.shape[2])):

        R = RR[:,:,nb_view].reshape(3,3)
        T = TT[:,[nb_view]]

        cam_direction = (R.T @ (np.array([0, 0, 1]).reshape(3, 1))).reshape(3)

        #fig = plt.figure()
        #ax = plt.axes(projection='3d')

        for face_id in range(normals.shape[1]):

            normal_id = normals[:,face_id]
            is_visible = np.dot(-cam_direction,normal_id)>1e-5
            normal_id = normal_id.reshape(3,1)
            if is_visible :

                dir_refracte = compute_refracted_ray_vecto(cam_direction.reshape(3,1),normal_id,1,1.56)
                vert = np.concatenate((vertices[:, faces[:, face_id]],np.ones((1,3))),axis=0)

                t = (-d[face_id] - np.sum(vox * normal_id,axis=0))/ np.sum(-dir_refracte * normal_id)
                intersection = vox - dir_refracte * t
                intersection = np.concatenate((intersection,np.ones((1,intersection.shape[1]))),axis=0)
                X = np.linalg.lstsq(vert,intersection,rcond=None)
                inside_face = np.sum(np.sign(X[0]),axis=0) == 3
                ind_inside = ind[inside_face]
                intersection_inside = intersection[:3,inside_face]
                intersection_outside = intersection[:3,np.logical_not(inside_face)]

                """
                fig = plt.figure()
                ax = plt.axes(projection='3d')
                ff = faces[:, face_id].tolist()
                ff = ff + [ff[0]]
                tri = vertices[:, ff]
                ax.plot(tri[0,:],tri[1,:],tri[2,:])
                ax.scatter(intersection_inside[0,:],intersection_inside[1,:],intersection_inside[2,:])
                ax.scatter(intersection_outside[0, :], intersection_outside[1, :], intersection_outside[2, :])
                plt.show()
                """


                vox_cam = R @ intersection_inside + T
                vox_cam[2,:] = 1.0
                vox_pixel = K @ vox_cam
                vox_pixel = vox_pixel[:2,:].astype(int)
                ind_inside_im = (vox_pixel[0,:]>=0) & (vox_pixel[0,:]<1920) & (vox_pixel[1,:]>=0) & (vox_pixel[1,:]<1080)

                vox_pixel = vox_pixel[:,ind_inside_im]
                vox_cam = vox_cam[:,ind_inside_im]
                ind_outside_im = ind_inside[np.logical_not(ind_inside_im)]
                ind_inside = ind_inside[ind_inside_im]

                voxel_in_mask = all_mask[nb_view,vox_pixel[1,:],vox_pixel[0,:]]
                ind_inside_mask = ind_inside[voxel_in_mask]
                ind_outside_mask = ind_inside[np.logical_not(voxel_in_mask)]
                count_positive[ind_inside_mask] +=1
                count_negative[ind_outside_mask] += 1
                count_negative[ind_outside_im] += 1
                vox_cam_inside = R.T @ (vox_cam[:,voxel_in_mask] - T)
                #ax.scatter(vox_cam_inside[0, :], vox_cam_inside[1, :], vox_cam_inside[2, :])
        #plt.show()

    probability = count_positive / np.maximum(count_positive+count_negative,np.ones(count_positive.shape))
    return probability,k



nbThread = os.cpu_count()//2
pack_voxels = np.array_split(voxels,nbThread,axis=1)
results = []
order = []

with concurrent.futures.ProcessPoolExecutor(max_workers=nbThread) as executor:
    threads = []
    for k in range(nbThread):
        print(k)
        threads.append(executor.submit(compute_SFS,pack_voxels[k],k))

    for f in concurrent.futures.as_completed(threads):
        probability,k = f.result()
        results.append(probability)
        order.append(k)

sort_index = np.argsort(order)
results = [results[i] for i in sort_index]
probability = np.hstack(results)



probability = probability.reshape(int(np.imag(n)),int(np.imag(n)),int(np.imag(n)))
data_grid = {"grid": probability,"XYZ":XYZ}
np.savez(folder+"grid.npz",**data_grid)
scipy.io.savemat(folder + 'grid.mat', data_grid)
a=0

