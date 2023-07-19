import copy
import pyvista as pv
import numpy as np
import math
import cv2
from tqdm import tqdm
from pyoctree import pyoctree as ot
import trimesh
import matplotlib.pyplot as plt

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


if __name__ == '__main__':


    folder = "D:\PhD\Projects\Blender_Data_Generation\OUTPUT\Data_xyz_normal/"

    mask = cv2.imread(folder+"mask_ref.png").sum(axis=2)
    im_normal = cv2.imread(folder+"mask_ref.png")
    pixels =  np.asarray(np.where(mask > 10))
    im_normal[np.where(mask<200)] = 128
    pixels = pixels[-1::-1, :]
    line = np.ones((1,pixels.shape[1]))
    pixels = np.concatenate((pixels,line),axis=0)
    pixels_im_ind = pixels[:2, :].T

    R = np.load(folder+"R.npy").reshape(3,3)
    T = np.load(folder+"T.npy")
    K = np.load(folder + "K.npy")
    cam_world = -R.T @ T
    cam_world_dir = R.T @ (np.array([0,0,1]).reshape(3,1)-T) - cam_world

    pixels_cam = np.linalg.inv(K) @ pixels
    pixels_world = R.T @ (pixels_cam - T)

    pathObj = "D:/INPUT/Graphosoma/Graphosoma.obj"
    mesh2 = trimesh.load(pathObj)

    rotation = trimesh.transformations.rotation_matrix(
        np.deg2rad(90), [1, 0, 0], point=[0, 0, 0])
    mesh2.apply_transform(rotation)

    interface = np.load(folder+"interface.npz")
    plan_normal = []
    plan_d = []
    plan_vertices = []
    for k in range(len(interface["faces"])):
        ff = interface["faces"][k].astype(int)
        vv = interface["vertices"][ff,:]*1.4
        nn = interface["normals"][k]
        if np.dot(nn,-cam_world_dir) > 1e-8 :
            dd = - np.mean(vv @ nn.reshape(3,1))
            plan_vertices.append(vv.T)
            plan_normal.append(nn)
            plan_d.append(dd)

        a=0

    origin = cam_world.reshape(3)
    all_normals = []
    all_intersection = []
    all_directions = []
    for k in tqdm(range(pixels_world.shape[1])):
        save = []
        direction = pixels_world[:, k]-origin
        all_directions.append(direction)
        for j in range(len(plan_normal)):
            normal = plan_normal[j]
            t = (-plan_d[j] - np.sum(origin * normal)) / np.sum(direction * normal)
            intersection = origin + direction * t
            coeff = np.linalg.lstsq(np.concatenate((plan_vertices[j],np.ones((1,3))),axis=0),np.concatenate((intersection.reshape(3,1),np.ones((1,1)))))
            if np.sum(np.sign(coeff[0])) == 3.0 :
                all_intersection.append(intersection)
                all_normals.append(normal)


    # test on a sphere mesh
    all_directions = np.array(all_directions)
    all_normals = np.array(all_normals)
    refracted_rays = compute_refracted_ray_vecto(all_directions.T,all_normals.T,1,1.33)
    refracted_rays = refracted_rays.T


    vertices = np.array(mesh2.vertices)
    faces = np.array(mesh2.faces).astype(int)

    normals = []
    octree = ot.PyOctree(vertices,faces)
    #a = cam_world.reshape(3)
    for k in tqdm(range(pixels_world.shape[1])):
        a = all_intersection[k]
        b = a+refracted_rays[k]
        ray = np.array([a,b]).astype(np.float32)
        result = octree.rayIntersection(ray)
        if len(result)!= 0 :
            f = octree.polyList[result[0].triLabel]
            coeff = np.linalg.lstsq(np.concatenate((f.vertices.T,np.ones((1,3))),axis=0),np.concatenate((result[0].p.reshape(3,1),np.ones((1,1))),axis=0))
            #normals.append(octree.polyList[result[0].triLabel].N)
            normals.append((mesh2.vertex_normals[faces[result[0].triLabel],:].T @ coeff[0]).reshape(3))
        else :
            normals.append([0.0,0.0,0.0])
        c=0

    a=0
    normals = np.array(normals)
    normals = 255 * (normals + 1) / 2.0
    pixels_im_ind = pixels_im_ind.astype(int)
    im_normal[pixels_im_ind[:, 1], pixels_im_ind[:, 0], :] = normals
    plt.imsave(folder + "normal_cmp_ref.png", im_normal)



