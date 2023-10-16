import copy
import os.path

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


    folder = "D:\PhD\Dropbox\Data\Data\Data_ortho_normal_sphere/"
    sub_folder = "render_3/"
    if not os.path.exists(folder+sub_folder):
        os.mkdir(folder+sub_folder)
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

    pathObj = "D:\PhD\Dropbox\Data\Data\models/Sphere/Sphere.obj"
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
        print('\033[93m' + "\nVIEW [{}/{}]\n".format(view+1,RR.shape[2]) + '\033[0m')
        print('\033[94m' + "\nPRECOMPUTATION RUNNING ...\n" + '\033[0m')
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
            u = refracted_rays[k]
            A = np.dot(u,u)
            B = 2*np.dot(a,u)
            C = np.dot(a,a)-(1.07/2)**2
            result2 = np.roots([A,B,C])
            if np.sum(np.iscomplex(result2)) == 0 :
                result2.sort()
            else :
                result2 = []

            #ray = np.array([a,b]).astype(np.float32)
            #result = octree.rayIntersection(ray)
            if len(result2)!= 0 :
                #f = octree.polyList[result[0].triLabel]
                #coeff = np.linalg.lstsq(np.concatenate((f.vertices.T,np.ones((1,3))),axis=0),np.concatenate((result[0].p.reshape(3,1),np.ones((1,1))),axis=0),rcond=None)
                #normals.append(octree.polyList[result[0].triLabel].N)
                pts = a + result2[0]*u
                n_pts = pts / np.linalg.norm(pts)
                #normals.append((mesh2.vertex_normals[faces[result[0].triLabel],:].T @ coeff[0]).reshape(3))
                normals.append(n_pts)
                #depth.append(result[0].p)
                depth.append(pts)
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
        nb_bounces = 3

        print('\033[94m' + "\nLIGHT SIMULATION RUNNING ...\n" + '\033[0m')

        for kl, light in enumerate(lights):
            light = light.reshape(3, 1)
            scene = Scene(mesh, light, light_intensity, ior_outside, ior_inside)
            scene.render_light_scattering(min_intensity=0.05)
            im = np.zeros((1080,1920,3))
            for nb_level,level in enumerate(tqdm(scene.illumination)):
                light_level = 0
                for polygon_prism_field in level:
                    result, direction = polygon_prism_field.is_intersecting_points_poly(depth)
                    ind_ok = np.where(result)[0]
                    pixels_im_ind_self = pixels_im_ind[ind_ok,:]
                    depth_ok = depth[:,ind_ok]
                    normals_ok = normals[ind_ok,:]
                    if nb_level == 0 :
                        if len(ind_ok) > 0 :
                            light_level += 1
                        for kll in range(depth_ok.shape[1]):
                            ttt = np.sum(normals_ok[kll,:].reshape(-1)*-polygon_prism_field.direction.reshape(-1))*polygon_prism_field.intensity
                            ttt = max(0,ttt)
                            im[pixels_im_ind_self[kll, 1], pixels_im_ind_self[kll, 0],:] = im[pixels_im_ind_self[kll, 1], pixels_im_ind_self[kll, 0]] + ttt
                    else :
                        ind_ok_ok = []
                        for kll in range(depth_ok.shape[1]):
                            d = depth_ok[:,kll].reshape(-1)
                            u = -polygon_prism_field.direction.reshape(-1)
                            A = np.dot(u, u)
                            B = 2 * np.dot(d, u)
                            C = np.dot(d, d) - (1.07 / 2) ** 2
                            result2 = np.roots([A, B, C])
                            is_visble = True
                            if np.sum(np.iscomplex(result2)) == 0:
                                if np.sum(result2>1e-3) !=0 :
                                    is_visble = False
                            if is_visble :
                                ind_ok_ok.append(ind_ok[kll])
                        ind_ok_ok = np.array(ind_ok_ok,dtype=int)

                        depth_ok_ok = depth[:,ind_ok_ok]
                        if len(ind_ok_ok) > 0 :
                            depth_projected = polygon_prism_field.intersecting_points_poly(depth_ok_ok)
                            path = polygon_prism_field.path
                            for polygon_prism_field_parent in path :
                                if len(ind_ok_ok) > 0 :
                                    u = polygon_prism_field_parent.direction
                                    min_dist = np.linalg.norm(np.cross(-depth_projected.T,u,axisa=1,axisb=0),axis=1)
                                    ind_ok_ok = ind_ok_ok[np.where(min_dist>(1.07/2))[0]]
                                    ind_red = np.where(min_dist>(1.07/2))[0]
                                    depth_projected = polygon_prism_field_parent.intersecting_points_poly(depth_projected[:,ind_red])

                            normals_ok = normals[ind_ok_ok,:]
                            pixels_im_ind_self = pixels_im_ind[ind_ok_ok, :]
                            if len(ind_ok_ok) > 0 :
                                light_level += 1
                            for kll in range(len(ind_ok_ok)):
                                ttt = np.sum(normals_ok[kll, :].reshape(-1) * -polygon_prism_field.direction.reshape(
                                    -1)) * polygon_prism_field.intensity
                                ttt = max(0, ttt)
                                im[pixels_im_ind_self[kll, 1], pixels_im_ind_self[kll, 0], :] = im[pixels_im_ind_self[
                                    kll, 1], pixels_im_ind_self[kll, 0]] + ttt
                print('\033[94m' + "\nKEPT [{}/{}] LIGHTS IN LEVEL {}\n".format(light_level,len(level),nb_level) + '\033[0m')
            im = im/(3*light_intensity)
            a = np.min(im)
            b = np.max(im)
            plt.imsave(folder+sub_folder+(3-len(str(view)))*"0"+str(view)+"_"+(3-len(str(kl)))*"0"+str(kl)+".png",im)




        normals = (normals + 1) / 2.0
        im_normal[pixels_im_ind[:, 1], pixels_im_ind[:, 0], :] = normals



