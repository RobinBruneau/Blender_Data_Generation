import copy
from tqdm import tqdm
import numpy as np
from shapely.geometry import Polygon as SPolygon
from matplotlib import pyplot as plt
import bisect

def get_t_max_and_id(l_val,val,dir,vect_val) :
    id = bisect.bisect(l_val, val) - 1
    if id+1 == len(l_val) :
        id -= 1
    if dir > 0 :
        t_max = dir*(l_val[id+1]-val)/vect_val
    else :
        t_max = dir*(val-l_val[id])/vect_val
    return id,t_max

def get_t_max_and_id_allin(l_val,vals,vals_id,dir,vect_val) :
    all_id = np.empty(len(vals))
    for k,val in enumerate(vals) :
        all_id[k] = bisect.bisect(l_val, val) - 1

    #all_id[all_id+1 == len(l_val)] = -1
    all_id = all_id.astype(int)
    all_id = vals_id
    if dir > 0 :
        t_max = dir*(l_val[all_id+1]-vals)/vect_val
    else :
        t_max = dir*(vals-l_val[all_id])/vect_val
    return all_id,t_max


def find_path_in_voxels(l_point,l_vect,use_point,box_min,box_max,nb_voxels_per_dim,grid_value,xyz):


    val_x = np.linspace(box_min[0], box_max[0], nb_voxels_per_dim[0] + 1)
    val_y = np.linspace(box_min[1], box_max[1], nb_voxels_per_dim[1] + 1)
    val_z = np.linspace(box_min[2], box_max[2], nb_voxels_per_dim[2] + 1)

    dx = (box_max[0] - box_min[0]) / nb_voxels_per_dim[0]
    dy = (box_max[1] - box_min[1]) / nb_voxels_per_dim[1]
    dz = (box_max[2] - box_min[2]) / nb_voxels_per_dim[2]

    all_ind = []
    for k in tqdm(range(l_point.shape[1]),position=0, leave=True):

        if use_point[k] :

            point = l_point[:,k]
            vect = l_vect[:,k]

            x = point[0]
            y = point[1]
            z = point[2]

            step_x = int(np.sign(vect[0]))
            step_y = int(np.sign(vect[1]))
            step_z = int(np.sign(vect[2]))

            if vect[0] != 0.0 :
                t_delta_x = abs(dx / vect[0])
            else :
                t_delta_x = np.inf
            if vect[1] != 0.0 :
                t_delta_y = abs(dy / vect[1])
            else :
                t_delta_y = np.inf
            if vect[2] != 0.0 :
                t_delta_z = abs(dz / vect[2])
            else :
                t_delta_z = np.inf

            x_id, t_max_x = get_t_max_and_id(val_x,x,step_x,vect[0])
            y_id, t_max_y = get_t_max_and_id(val_y, y, step_y,vect[1])
            z_id, t_max_z = get_t_max_and_id(val_z, z, step_z,vect[2])


            ind = 0
            can_continue = True
            while can_continue :
                if t_max_x < t_max_y :
                    if t_max_x < t_max_z :
                        x_id += step_x
                        t_max_x += t_delta_x
                    else :
                        z_id += step_z
                        t_max_z += t_delta_z
                else :
                    if t_max_y < t_max_z :
                        y_id += step_y
                        t_max_y += t_delta_y
                    else :
                        z_id += step_z
                        t_max_z += t_delta_z

                if x_id < 0 or x_id >= nb_voxels_per_dim[0] or y_id < 0 or y_id >= nb_voxels_per_dim[1] or z_id < 0 or z_id >= nb_voxels_per_dim[2]:
                    can_continue = False
                else :
                    if grid_value[x_id,y_id,z_id] :
                        ind += 1

            all_ind.append(ind)
        else :
            all_ind.append(-1)
    return np.array(all_ind)


def find_path_in_voxels_allin(l_point,l_point_id,vect,use_point,box_min,box_max,nb_voxels_per_dim,grid_value,xyz):

    ind_use = np.where(use_point)[0]
    l_point = l_point[:,use_point]
    l_point_id = l_point_id[:,use_point]

    sx,sy,sz = nb_voxels_per_dim

    big_grid = np.zeros((sx * 3, sy * 3, sz * 3),dtype=bool)
    big_grid[sx:2*sx,sy:2*sy,sz:2*sz] = grid_value

    #big_grid_inside = np.zeros((sx * 3, sy * 3, sz * 3),dtype=bool)
    #big_grid_inside[sx:2*sx,sy:2*sy,sz:2*sz] = True


    val_x = np.linspace(box_min[0], box_max[0], nb_voxels_per_dim[0] + 1)
    val_y = np.linspace(box_min[1], box_max[1], nb_voxels_per_dim[1] + 1)
    val_z = np.linspace(box_min[2], box_max[2], nb_voxels_per_dim[2] + 1)

    dx = (box_max[0] - box_min[0]) / nb_voxels_per_dim[0]
    dy = (box_max[1] - box_min[1]) / nb_voxels_per_dim[1]
    dz = (box_max[2] - box_min[2]) / nb_voxels_per_dim[2]


    step = np.sign(vect).astype(int)

    if vect[0] != 0.0:
        t_delta_x = abs(dx / vect[0])
    else:
        t_delta_x = np.inf
    if vect[1] != 0.0:
        t_delta_y = abs(dy / vect[1])
    else:
        t_delta_y = np.inf
    if vect[2] != 0.0:
        t_delta_z = abs(dz / vect[2])
    else:
        t_delta_z = np.inf

    #x_id, t_max_x = get_t_max_and_id_allin(val_x, l_point[0,:],l_point_id[0,:], step[0], vect[0])
    #y_id, t_max_y = get_t_max_and_id_allin(val_y, l_point[1,:], step[1], vect[1])
    #z_id, t_max_z = get_t_max_and_id_allin(val_z, l_point[2,:], step[2], vect[2])
    x_id = l_point_id[0,:]
    y_id = l_point_id[1, :]
    z_id = l_point_id[2, :]

    t_max_x = t_delta_x
    t_max_y = t_delta_y
    t_max_z = t_delta_z

    all_ind = np.zeros((l_point.shape[1]))

    all_rem = np.empty((3,len(x_id)))
    if step[0] == 1 :
        all_rem[0,:] = sx-x_id
    else :
        all_rem[0,:] = x_id+1
    if step[1] == 1 :
        all_rem[1,:] = sy-y_id
    else :
        all_rem[1,:] = y_id+1
    if step[2] == 1 :
        all_rem[2,:] = sz-z_id
    else :
        all_rem[2,:] = z_id+1

    rem = all_rem.min(axis=0)
    rem_ind = all_rem.argmin(axis=0)
    ix = rem[rem_ind == 0]
    iy = rem[rem_ind == 1]
    iz = rem[rem_ind == 2]
    if len(ix) == 0 :
        rem_x = 0
    else :
        rem_x = np.max(ix)
    if len(iy) == 0 :
        rem_y = 0
    else :
        rem_y = np.max(iy)
    if len(iz) == 0 :
        rem_z = 0
    else :
        rem_z = np.max(iz)

    #print("")
    #print(step)
    #print(rem_x,rem_y,rem_z)
    can_continue = True
    while can_continue:

        if t_max_x < t_max_y:
            if t_max_x < t_max_z:
                x_id += step[0]
                rem_x -= 1
                t_max_x += t_delta_x
            else:
                z_id += step[2]
                rem_z -= 1
                t_max_z += t_delta_z
        else:
            if t_max_y < t_max_z:
                y_id += step[1]
                rem_y -= 1
                t_max_y += t_delta_y
            else:
                z_id += step[2]
                rem_z -= 1
                t_max_z += t_delta_z

        all_ind = all_ind + big_grid[sx+x_id,sy+y_id,sz+z_id]
        #sum_up = np.sum(big_grid_inside[sx+x_id,sy+y_id,sz+z_id])
        can_continue = np.sum(np.sign([rem_x,rem_y,rem_z])) != -3.0

    #print(rem_x,rem_y,rem_z)
    ll_all_ind = -1*np.ones((len(use_point)))
    ll_all_ind[ind_use] = all_ind

    return ll_all_ind,all_ind


def snell_refraction(incident,n,ior_1,ior_2):

    incident = normalize(incident)
    n = normalize(-n)

    mu = ior_1/ior_2
    ni = n.T@incident
    refracted = mu*(incident-n*ni) + n*np.sqrt(1-(mu**2)*(1-ni**2))

    return refracted

def fresnel_coefficient(n1,n2,n,i):
    ni = n.T@i
    mu = n1/n2

    if ni**2>= 1-1/(mu**2) :
        Rs = (mu*ni - np.sqrt(1-(mu**2)*(1-ni**2)))**2 / (mu*ni + np.sqrt(1-(mu**2)*(1-ni**2)))**2
        Re = (mu*np.sqrt(1-(mu**2)*(1-ni**2)) - ni)**2 / (mu*np.sqrt(1-(mu**2)*(1-ni**2)) + ni)**2
    else :
        Rs = Re = 1

    R = float(0.5*(Rs+Re))
    T = 1-R
    return R,T

def normalize(v):
    return v/np.linalg.norm(v)


class Scene():

    def __init__(self,mesh,directional_light,directional_light_intensity,ior_outside,ior_inside):

        self.mesh = mesh
        self.directional_light = directional_light
        self.directional_light_intensity = directional_light_intensity
        self.ior_outside = ior_outside
        self.ior_inside = ior_inside
        self.illumination = []

    def render_light_scattering(self,nb_bounces=0,min_intensity=0):

        assert(nb_bounces>=0)
        if min_intensity > 0 :
            self.render_light_scattering_min_intensity(min_intensity)
        else :
            self.render_light_scattering_max_bounces(nb_bounces)

    def render_light_scattering_max_bounces(self,nb_bounces):

        levels_triangular_prism_field = []
        current_level = []
        # Initial Snell's Transmission :
        for polygon in self.mesh :
            if polygon.n.T@-self.directional_light > 0 :

                refracted_direction = snell_refraction(self.directional_light, polygon.n, self.ior_outside, self.ior_inside)
                _, T = fresnel_coefficient(self.ior_outside, self.ior_inside, polygon.n, -self.directional_light)
                current_level.append(Polygon_prism_field(polygon.points,refracted_direction,0,self.directional_light_intensity*T,polygon.n))
        levels_triangular_prism_field.append(copy.deepcopy(current_level))

        for bounce in range(nb_bounces) :
            next_level = []
            for polygon_prism in current_level:
                result_bounce = polygon_prism.projection_on_mesh(self.mesh,self.ior_inside,self.ior_outside)
                if len(result_bounce)>0:
                    next_level.append(result_bounce)
            current_level = [item for sublist in next_level for item in sublist]
            levels_triangular_prism_field.append(current_level)

        self.illumination = levels_triangular_prism_field

    def render_light_scattering_min_intensity(self, min_intensity):

        levels_triangular_prism_field = []
        current_level = []
        # Initial Snell's Transmission :
        for polygon in self.mesh:
            if polygon.n.T @ -self.directional_light > 0:
                refracted_direction = snell_refraction(self.directional_light, polygon.n, self.ior_outside,
                                                       self.ior_inside)
                _, T = fresnel_coefficient(self.ior_outside, self.ior_inside, polygon.n, -self.directional_light)
                new_intensity = self.directional_light_intensity * T
                if new_intensity > min_intensity :
                    current_level.append(Polygon_prism_field(polygon.points, refracted_direction, 0, new_intensity,polygon.n))
        levels_triangular_prism_field.append(copy.deepcopy(current_level))

        while(len(current_level)>0):
            next_level = []
            for polygon_prism in current_level:
                result_bounce = polygon_prism.projection_on_mesh(self.mesh, self.ior_inside, self.ior_outside)
                if len(result_bounce) > 0:
                    next_level.append(result_bounce)
            current_level_possibilities = [item for sublist in next_level for item in sublist]
            current_level = []
            for tri_ray in current_level_possibilities :
                if tri_ray.intensity > min_intensity :
                    current_level.append(tri_ray)
            if len(current_level) > 0 :
                levels_triangular_prism_field.append(current_level)

        self.illumination = levels_triangular_prism_field

    def query_quivers_field(self,x):

        # Check inside mesh
        is_inside = True
        for polygon in self.mesh :
            is_inside *= ((x-polygon.points[:,[0]]).T @ polygon.n) <0

        quivers_field = []
        if is_inside :
            for level in self.illumination :
                for polygon_prism_field in level :
                    result = polygon_prism_field.is_intersecting_point(x)
                    if result :
                        quivers_field.append(polygon_prism_field)

            return True,quivers_field
        else :
         return False,[]

    def query_quivers_field_vectorize(self,x,x_id,min_xyz,max_xyz,size_grid,grid,xyz):
        quivers_fields = [np.zeros((3,1)) for k in range(x.shape[1])]
        nb_lights = [0 for k in range(x.shape[1])]
        all_lights_visu = []
        all_lights_visu_remove= []
        all_lights_intensity = []
        for level in self.illumination:
            for polygon_prism_field in level:
                #result,direction = polygon_prism_field.is_intersecting_points_tri(x)
                result, direction = polygon_prism_field.is_intersecting_points_poly(x)
                all_lights_visu.append(np.array(result))
                l_dir = np.repeat(-direction,x.shape[1],axis=1)
                #nb_intersection = find_path_in_voxels(x, l_dir, result, min_xyz, max_xyz, size_grid, grid,xyz)
                nb_intersection,nb_intersection_visible = find_path_in_voxels_allin(x,x_id,-direction,result,min_xyz,max_xyz,size_grid,grid,xyz)
                #plt.figure()
                #plt.hist(nb_intersection_visible,bins=20)
                #plt.show()
                nb_intersection = np.where(nb_intersection<10,1,0)
                #all_lights_visu_remove.append(np.array(np.logical_and(result,nb_intersection)))
                all_lights_intensity.append([polygon_prism_field.intensity,polygon_prism_field.n])
                for k in range(len(result)):
                    if result[k]:
                        quivers_fields[k]+=polygon_prism_field.direction*polygon_prism_field.intensity*nb_intersection[k]
                        #nb_lights[k]+=polygon_prism_field.intensity*nb_intersection[k]
        nb_lights = np.array(nb_lights)
        return quivers_fields,nb_lights,all_lights_visu,all_lights_visu_remove,all_lights_intensity


class Polygon():

    def __init__(self,points):

        assert(len(points.shape) == 2)
        assert(np.sum((points[:,0]-points[:,-1]))<1e-8)

        self.points = points
        self.d1 = normalize(points[:,[1]]-points[:,[0]])
        self.d2 = normalize(points[:,[-2]]-points[:,[0]])
        self.n = normalize(np.cross(self.d1,self.d2,axis=0))
        self.polygon_to_world_matrix = np.concatenate((self.d1,self.d2,self.n),axis=1)
        self.beta = -self.n.T@self.points[:,[0]]
        self.world_to_polygon_matrix = np.linalg.inv(self.polygon_to_world_matrix)
        self.points_in_polygon_base = self.world_to_polygon_matrix @ self.points
        self.polygon_in_polygon_base = SPolygon((self.points_in_polygon_base[:2,:].T).tolist())


class Polygon_prism_field():

    def __init__(self,polygon_base,direction,bounce_level,intensity,n,path=[]):

        self.polygon_base = polygon_base
        self.direction = direction
        self.bounce_level = bounce_level
        self.intensity = intensity
        self.path = path
        self.n = n
        self.beta = -self.n.T @ self.polygon_base[:, [0]]
        self.bounding_box = np.array([[np.min(polygon_base[0,:]),np.max(polygon_base[0,:])],
                                      [np.min(polygon_base[1,:]),np.max(polygon_base[1,:])],
                                      [np.min(polygon_base[2,:]),np.max(polygon_base[2,:])]])

    def projection_on_3D_polygon(self,polygon,n1,n2):

        if self.direction.T@polygon.n > 0 :

            projected_prism = (np.eye(3) - (self.direction@polygon.n.T)/(self.direction.T@polygon.n)) @ self.polygon_base - (polygon.beta*self.direction)/(self.direction.T@polygon.n)
            projected_prism_in_polygon_base = polygon.world_to_polygon_matrix@projected_prism
            projected_3rd_dim_value = np.mean(projected_prism_in_polygon_base[2,:])
            polygon_projected_prism_in_polygon_base = SPolygon((projected_prism_in_polygon_base[:2,:].T).tolist())
            intersection = shapely.intersection(polygon_projected_prism_in_polygon_base, polygon.polygon_in_polygon_base)
            if intersection.is_empty :
                return []
            else :
                if intersection.geom_type != 'Polygon' :
                    return []
                else :

                    intersection_coordinates = np.array([[i[0], i[1]] for i in list(intersection.exterior.coords)])

                    R,_ = fresnel_coefficient(n1,n2,polygon.n,self.direction)
                    new_intensity = R * self.intensity
                    new_direction = self.direction-2*polygon.n*(self.direction.T@polygon.n)
                    polygon_3d = np.concatenate((intersection_coordinates.T,projected_3rd_dim_value*np.ones((1,intersection_coordinates.shape[0]))),axis=0)
                    polygon_base_in_world = polygon.polygon_to_world_matrix @ polygon_3d
                    l_triangular_prism_field_after_bounce = [Polygon_prism_field(polygon_base_in_world,new_direction,self.bounce_level+1,new_intensity,polygon.n,path=self.path+[self])]
                    return l_triangular_prism_field_after_bounce
        else :
            return []

    def projection_on_mesh(self,mesh,n1,n2):

        all_l_polygon_prism_after_bounce = []
        for polygon in mesh :
            result_bounce = self.projection_on_3D_polygon(polygon,n1,n2)
            if len(result_bounce) > 0 :
                all_l_polygon_prism_after_bounce = all_l_polygon_prism_after_bounce + result_bounce
        return all_l_polygon_prism_after_bounce

    def is_intersecting_point(self,x):

        A = np.eye(3) - (-self.direction @ self.n.T) / (-self.direction.T @ self.n)
        B = (self.beta * -self.direction) / (-self.direction.T @ self.n)
        projected_x = A @ x - B

        #projected_x = (np.eye(3) - (-self.direction @ self.n.T) / (
        #        -self.direction.T @ self.n)) @ x - (self.beta * -self.direction) / (
        #                          -self.direction.T @ self.n)
        # check inside bounding box
        if (self.bounding_box[:,[0]]-projected_x).T @ (self.bounding_box[:,[1]]-projected_x) <= 0 :
           v = self.polygon_base - projected_x
           print(self.polygon_base.shape)
           v = v/np.linalg.norm(v,axis=0)
           angles = np.arccos(np.clip(np.sum(v[:,:-1]*v[:,1:],axis=0),-1,1))
           if abs(sum(angles)-2*np.pi)<1e-5 :
               return True
           else :
               return False
        else :
            return False

    def is_intersecting_points_tri(self,x):

        A = np.eye(3) - (-self.direction @ self.n.T) / (-self.direction.T @ self.n)
        B = (self.beta * -self.direction) / (-self.direction.T @ self.n)
        projected_x = A@x-B

        AA = self.polygon_base[:,:3]
        AA = np.concatenate((AA,np.ones((1,3))),axis=0)
        projected_x = np.concatenate((projected_x,np.ones((1,projected_x.shape[1]))),axis=0)
        W = np.linalg.lstsq(AA,projected_x,rcond=None)[0]
        is_inside = np.sum(np.sign(W),axis=0) == 3
        return is_inside,self.direction


    def is_intersecting_points_poly(self,x):

        A = np.eye(3) - (-self.direction @ self.n.T) / (-self.direction.T @ self.n)
        B = (self.beta * -self.direction) / (-self.direction.T @ self.n)
        projected_x = A@x-B

        vv = np.empty((3,x.shape[1],self.polygon_base.shape[1]))
        for k in range(self.polygon_base.shape[1]):
            vv[:,:,k] = self.polygon_base[:,[k]]-projected_x
        vv = vv / np.linalg.norm(vv,axis=0,keepdims=True)
        angles = np.arccos(np.clip(np.sum(vv[:,:, :-1] * vv[:,:, 1:], axis=0), -1, 1))
        is_inside = np.abs(np.sum(angles,axis=1) - 2 * np.pi) < 1e-5
        return is_inside,self.direction


def generate_mesh(vertices,faces) :

    assert(vertices.shape[1] == 3)
    assert(faces.shape[1] >= 3)

    faces = np.concatenate((faces,faces[:,[0]]),axis=1)
    mesh = []
    for face in faces :
        points = (vertices[face,:]).T
        mesh.append(Polygon(points))

    return mesh





