import copy
import numpy as np
import math
from tqdm import tqdm
from pyoctree import pyoctree as ot
import trimesh
import matplotlib.pyplot as plt


def ray_box_intersection(ray_origin, ray_direction, box_min, box_max):
    t_near = -np.inf
    t_far = np.inf

    for i in range(3):
        # Check for parallel rays
        if abs(ray_direction[i]) < 1e-6:
            if ray_origin[i] < box_min[i] or ray_origin[i] > box_max[i]:
                return False,None, None  # No intersection, no normal
        else:
            t1 = (box_min[i] - ray_origin[i]) / ray_direction[i]
            t2 = (box_max[i] - ray_origin[i]) / ray_direction[i]
            t_near = max(t_near, min(t1, t2))
            t_far = min(t_far, max(t1, t2))

    if t_near > t_far or t_far < 0:
        return False,None, None  # No intersection, no normal

    intersection_point = ray_origin + t_near * ray_direction

    for i in range(3):
        if t_near == (box_min[i] - ray_origin[i]) / ray_direction[i]:
            normal = np.zeros(3)
            normal[i] = -1
            return True,intersection_point, normal

        if t_near == (box_max[i] - ray_origin[i]) / ray_direction[i]:
            normal = np.zeros(3)
            normal[i] = 1
            return True,intersection_point, normal

    return False,intersection_point, None


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


    folder = "D:/PhD/Projects/Playing_with_NeuS/data/Graphosoma_RMVS_50_2/"
    RR = np.load(folder + "R.npy")
    TT = np.load(folder + "T.npy")
    K = np.load(folder + "K.npy")

    pathObj = "D:/PhD/Dropbox/Data/Data/models/Graphosoma/Graphosoma.obj"
    mesh2 = trimesh.load(pathObj)

    rotation = trimesh.transformations.rotation_matrix(
        np.deg2rad(90), [1, 0, 0], point=[0, 0, 0])
    mesh2.apply_transform(rotation)
    vertices = np.array(mesh2.vertices)
    faces = np.array(mesh2.faces).astype(np.int32)
    octree = ot.PyOctree(vertices, faces)

    interface_data = np.load(folder + "interface.npz")
    vertices_int = interface_data["vertices"].T
    box_min = np.min(vertices_int, axis=0)
    box_max = np.max(vertices_int, axis=0)
    centers_int = interface_data["centers"].T
    faces_int = (interface_data["faces"].T).astype(int)
    normals_int = (interface_data["normals"] / np.linalg.norm(interface_data["normals"], axis=0)).T

    a=0
    for view in range(29,RR.shape[2]):

        R = RR[:,:,view].reshape(3,3)
        T = TT[:,[view]]

        mask = plt.imread(folder+"mask/{}.png".format((3-len(str(view)))*"0"+str(view)))[:,:,:3].sum(axis=2)
        im_normal = plt.imread(folder+"mask/{}.png".format((3-len(str(view)))*"0"+str(view)))[:,:,:3]
        pixels =  np.asarray(np.where(mask > 0.1))
        im_normal[np.where(mask<0.1)] = 0.5
        pixels = pixels[-1::-1, :]
        line = np.ones((1,pixels.shape[1]))
        pixels = np.concatenate((pixels,line),axis=0)
        pixels_im_ind = pixels[:2, :].T


        cam_world = -R.T @ T
        cam_world_dir = R.T @ (np.array([0,0,1]).reshape(3,1)-T) - cam_world

        pixels_cam = np.linalg.inv(K) @ pixels
        pixels_world = R.T @ (pixels_cam - T)


        origin = cam_world.reshape(3)
        all_normals = []
        all_intersection = []
        all_directions = []
        all_is_intersecting = []
        for k in tqdm(range(pixels_world.shape[1])):
            direction = pixels_world[:, k]-origin
            is_intersecting,intersection,normal = ray_box_intersection(cam_world.reshape(3),direction,box_min,box_max)
            all_is_intersecting.append(is_intersecting)
            if is_intersecting :
                all_normals.append(normal)
                all_directions.append(direction)
                all_intersection.append(intersection)
            else :
                all_normals.append([0.0,0.0,0.0])
                all_directions.append(direction)
                all_intersection.append([0.0,0.0,0.0])


        # test on a sphere mesh
        all_directions = np.array(all_directions)
        all_normals = np.array(all_normals)
        print(all_directions.shape,all_normals.shape)
        refracted_rays = compute_refracted_ray_vecto(all_directions.T,all_normals.T,1,1.56)
        refracted_rays = refracted_rays.T



        normals = []

        #a = cam_world.reshape(3)
        for k in tqdm(range(pixels_world.shape[1])):
            if all_is_intersecting[k] :
                a = all_intersection[k]
                b = a+refracted_rays[k]
                ray = np.array([a,b]).astype(np.float32)
                result = octree.rayIntersection(ray)
                if len(result)!= 0 :
                    f = octree.polyList[result[0].triLabel]
                    coeff = np.linalg.lstsq(np.concatenate((f.vertices.T,np.ones((1,3))),axis=0),np.concatenate((result[0].p.reshape(3,1),np.ones((1,1))),axis=0),rcond=None)
                    #normals.append(octree.polyList[result[0].triLabel].N)
                    normals.append((mesh2.vertex_normals[faces[result[0].triLabel],:].T @ coeff[0]).reshape(3))
                else :
                    normals.append([0.0,0.0,0.0])
            else :
                normals.append([0.0, 0.0, 0.0])

        a=0
        normals = np.array(normals)
        normals = (normals + 1) / 2.0
        pixels_im_ind = pixels_im_ind.astype(int)
        im_normal[pixels_im_ind[:, 1], pixels_im_ind[:, 0], :] = normals
        plt.imsave(folder + "normals/{}.png".format((3-len(str(view)))*"0"+str(view)), im_normal)



