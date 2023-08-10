import copy
import numpy as np
import math
from tqdm import tqdm
from pyoctree import pyoctree as ot
import trimesh
import matplotlib.pyplot as plt
from directional_3d_scene_polygonale_ORTHO_REAL import *
from functools import partial
tqdm = partial(tqdm, position=0, leave=True)

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


    folder = "/home/robin/Desktop/PhD/Data//Data_ortho_normal_full_2/"
    RR = np.load(folder + "R.npy")
    TT = np.load(folder + "T.npy")
    K = np.load(folder + "K.npy")

    vertices = 0.7 * np.array([[-1, 1, -1],
                               [1, 1, -1],
                               [-1, 1, 1],
                               [1, 1, 1],
                               [-1, -1, 1],
                               [1, -1, 1],
                               [-1, -1, -1],
                               [1, -1, -1]]).T

    faces = np.array([[2, 3, 1, 0],
                      [4, 5, 3, 2],
                      [6, 7, 5, 4],
                      [0, 1, 7, 6],
                      [6, 4, 2, 0],
                      [5, 7, 1, 3]]).astype(int).T

    mesh = generate_mesh(vertices.T, faces.T)

    lights_data = np.load(folder + "lights.npz")
    nb_view = len(lights_data.files) // 2
    nb_light = lights_data["cam_0"].shape[0]

    pathObj = "/home/robin/Desktop/PhD/Data/Graphosoma/Graphosoma.obj"
    mesh2 = trimesh.load(pathObj)

    rotation = trimesh.transformations.rotation_matrix(
        np.deg2rad(90), [1, 0, 0], point=[0, 0, 0])
    mesh2.apply_transform(rotation)
    vertices = np.array(mesh2.vertices)
    faces = np.array(mesh2.faces).astype(np.int32)

    interface_old = np.load(folder + "interface.npz")
    interface = {}
    for e in interface_old.files :
        interface.update({e:interface_old[e].T})

    octree = ot.PyOctree(vertices, faces)

    a=0
    for view in range(0,RR.shape[2]):

        R = RR[:,:,view].reshape(3,3)
        T = TT[:,[view]]

        mask = plt.imread(folder+"mask/{}.png".format((3-len(str(view)))*"0"+str(view)))[:,:,:3].sum(axis=2)
        im_normal = plt.imread(folder+"mask/{}.png".format((3-len(str(view)))*"0"+str(view)))[:,:,:3]
        pixels =  np.asarray(np.where(mask > 0.1))
        im_normal[np.where(mask<0.1)] = 0.5
        pixels = pixels[-1::-1, :]
        line = np.ones((1,pixels.shape[1]))
        pixels = np.concatenate((pixels,line),axis=0)
        pixels_im_ind = pixels[:2, :].T.astype(int)


        cam_direction = (R.T @ (np.array([0,0,1]).reshape(3,1))).reshape(3)

        pixels_cam = np.linalg.inv(K) @ pixels
        pixels_world = R.T @ (pixels_cam - T)


        plan_normal = []
        plan_d = []
        plan_vertices = []
        for k in range(len(interface["faces"])):
            ff = interface["faces"][k].astype(int)
            vv = interface["vertices"][ff,:]
            nn = interface["normals"][k]
            nn = nn//np.linalg.norm(nn)
            if np.dot(nn,-cam_direction) > 1e-8 :
                dd = - np.mean(vv @ nn.reshape(3,1))
                plan_vertices.append(vv.T)
                plan_normal.append(nn)
                plan_d.append(dd)

        all_normals = []
        all_intersection = []
        all_directions = []
        for k in tqdm(range(pixels_world.shape[1])):
            save = []
            origin = pixels_world[:, k]
            all_directions.append(cam_direction)
            done = False
            for j in range(len(plan_normal)):
                normal = plan_normal[j]
                t = (-plan_d[j] - np.sum(origin * normal)) / np.sum(cam_direction * normal)
                intersection = origin + cam_direction * t
                coeff = np.linalg.lstsq(np.concatenate((plan_vertices[j],np.ones((1,3))),axis=0),np.concatenate((intersection.reshape(3,1),np.ones((1,1)))),rcond=None)
                if np.sum(np.sign(coeff[0])) == 3.0 :
                    if not done :
                        all_intersection.append(intersection)
                        all_normals.append(normal)
                        done = True


        # test on a sphere mesh
        all_directions = np.array(all_directions)
        all_normals = np.array(all_normals)
        refracted_rays = compute_refracted_ray_vecto(all_directions.T,all_normals.T,1,1.56)
        refracted_rays = refracted_rays.T



        normals = []
        depth = []

        #a = cam_world.reshape(3)
        for k in tqdm(range(pixels_world.shape[1])):
            a = all_intersection[k]
            b = a+refracted_rays[k]
            ray = np.array([a,b]).astype(np.float32)
            result = octree.rayIntersection(ray)
            if len(result)!= 0 :
                f = octree.polyList[result[0].triLabel]
                coeff = np.linalg.lstsq(np.concatenate((f.vertices.T,np.ones((1,3))),axis=0),np.concatenate((result[0].p.reshape(3,1),np.ones((1,1))),axis=0),rcond=None)
                #normals.append(octree.polyList[result[0].triLabel].N)
                normals.append((mesh2.vertex_normals[faces[result[0].triLabel],:].T @ coeff[0]).reshape(3))
                depth.append(result[0].p)
            else :
                normals.append([0.0,0.0,0.0])
                depth.append([0.0,0.0,0.0])
            c=0

        a=0
        normals = np.array(normals)
        depth = np.array(depth).T

        lights = lights_data["cam_{}".format(view)]
        lights = lights / np.linalg.norm(lights, axis=1, keepdims=True)
        light_intensity = np.pi
        ior_outside = 1.0
        ior_inside = 1.56
        nb_bounces = 0
        for kl, light in enumerate(tqdm(lights)):
            light = light.reshape(3, 1)
            scene = Scene(mesh, light, light_intensity, ior_outside, ior_inside)
            scene.render_light_scattering(nb_bounces=nb_bounces)  # ,min_intensity=0.05)
            im = np.zeros((1080,1920,3))
            for level in scene.illumination:
                for polygon_prism_field in level:
                    result, direction = polygon_prism_field.is_intersecting_points_poly(depth)
                    ind_ok = np.where(result)[0]
                    pixels_im_ind_self = pixels_im_ind[ind_ok,:]
                    depth_ok = depth[:,ind_ok]
                    normals_ok = normals[ind_ok,:]
                    for kll in tqdm(range(depth_ok.shape[1])):
                        a = depth_ok[:,kll]
                        b = a - polygon_prism_field.direction.reshape(-1)
                        ray = np.array([a, b]).astype(np.float32)
                        result = octree.rayIntersection(ray)
                        is_ok_ray = True
                        if len(result) != 0:
                            if result[-1].s >=1e-3 :
                                is_ok_ray = False

                        if is_ok_ray :
                            ttt = np.sum(normals_ok[kll,:].reshape(-1)*-polygon_prism_field.direction.reshape(-1))*polygon_prism_field.intensity
                            ttt = max(0,ttt)
                            im[pixels_im_ind_self[kll, 1], pixels_im_ind_self[kll, 0],:] = im[pixels_im_ind_self[kll, 1], pixels_im_ind_self[kll, 0]] + ttt

            im = im/(3*light_intensity)
            a = np.min(im)
            b = np.max(im)
            plt.imsave(folder+"render/"+(3-len(str(view)))*"0"+str(view)+"_"+(3-len(str(kl)))*"0"+str(kl)+".png",im)




        normals = (normals + 1) / 2.0
        im_normal[pixels_im_ind[:, 1], pixels_im_ind[:, 0], :] = normals
        #plt.imsave(folder + "normal/{}.png".format((3-len(str(view)))*"0"+str(view)), im_normal)



