import numpy as np

def generate_points_on_sphere(radius,number_points):
    points = []
    a = 4 * np.pi / number_points
    d = np.sqrt(a)
    Mv = np.round(np.pi / d).astype(int)
    dv = np.pi / Mv
    dphi = a / dv
    for m in range(Mv):
        v = np.pi * (m + 0.5) / Mv
        Mphi = np.round(2 * np.pi * np.sin(v) / dphi).astype(int)
        for n in range(Mphi):
            phi = 2 * np.pi * n / Mphi
            location = [radius * np.sin(v) * np.cos(phi), radius * np.sin(v) * np.sin(phi), radius * np.cos(v)]
            points.append(location)
    points = np.array(points).T
    random_rotation = euler_to_matrix(np.random.random(3)*2*np.pi)
    points = random_rotation @ points
    return points.T

def euler_to_matrix(rotation):
    Rx = np.array([[1, 0, 0],
                   [0,np.cos(rotation[0]),-np.sin(rotation[0])],
                   [0,np.sin(rotation[0]),np.cos(rotation[0])]])

    Ry = np.array([[np.cos(rotation[1]), 0, np.sin(rotation[1])],
                   [0, 1, 0],
                   [-np.sin(rotation[1]), 0, np.cos(rotation[1])]])

    Rz = np.array([[np.cos(rotation[2]), -np.sin(rotation[2]),0],
                   [np.sin(rotation[2]), np.cos(rotation[2]),0],
                   [0, 0, 1 ]])
    return Rz@Ry@Rx

class Camera():

    def __init__(self,id,location,rotation,lens,is_looking_at,looking_at):

        self.id = id
        self.location = location
        self.rotation = rotation
        self.lens = lens
        self.is_looking_at = is_looking_at
        self.looking_at = looking_at

    def get_looking_direction(self):
        if self.is_looking_at :
            d = self.looking_at-self.location
            return (d/np.linalg.norm(d)).reshape((3,1))
        else :
            return euler_to_matrix(self.rotation) @ np.array([[0],[0],[-1]])

class CameraManager():

    def __init__(self):
        self.cameras = []

    def clean_cameras(self):
        self.cameras = []

    def ring_cameras(self,height,radius,number_cameras,lens):
        self.clean_cameras()
        angles = np.linspace(0,2*np.pi,number_cameras+1)[:-1]
        for k in range(len(angles)):
            location = np.array([radius*np.cos(angles[k]),radius*np.sin(angles[k]),height])
            c = Camera(id=0, location=location, rotation=None, lens=lens, is_looking_at=True, looking_at=np.array([0.0,0.0,0.0]))
            self.cameras.append(c)

    def sphere_cameras(self,radius,number_cameras,lens):
        self.clean_cameras()
        locations = generate_points_on_sphere(radius,number_cameras)
        for k,location in enumerate(locations) :
            c = Camera(id=k, location=np.array(location), rotation=None, lens=lens, is_looking_at=True,
                       looking_at=np.array([0.0, 0.0, 0.0]))
            self.cameras.append(c)

    def single_camera(self,location,rotation,lens,is_looking_at=False,looking_at=None):
        self.clean_cameras()
        c = Camera(id=0,location=location,rotation=rotation,lens=lens,is_looking_at=is_looking_at,looking_at=looking_at)
        self.cameras.append(c)

    def multi_cameras(self,locations,rotations,lens,is_looking_at=False,looking_at=None):
        self.clean_cameras()
        for k in range(len(locations)):
            c = Camera(id=k,location=locations[k],rotation=rotations[k],lens=lens,is_looking_at=is_looking_at,looking_at=looking_at)
            self.cameras.append(c)

class Light():

    def __init__(self,id,type,direction,color,strength):

        self.id = id
        self.type = type
        self.direction = direction
        self.color = color
        self.strength = strength

class LightManager():

    def __init__(self):
        self.fixed_light = None
        self.lights = []

    def clean_lights(self):
        self.fixed_light = None
        self.lights = []

    def fixed_ambiant(self,color,strength):
        self.clean_lights()
        l = Light(id="0",type="world",direction=None,color=color,strength=strength)
        self.lights.append(l)
        self.fixed_light = True

    def fixed_directionnals(self,directions,colors,strengths):
        self.clean_lights()
        for k in range(len(directions)):
            l = Light(id=str(k),type="sun", direction=directions[k], color=colors[k], strength=strengths[k])
            self.lights.append(l)
        self.fixed_light = True

    def semi_sphere_directionnal_per_camera(self,cameras,number_lights,colors,strengths,max_angle=80):
        self.clean_lights()
        for k,cam in enumerate(cameras) :
            cam_lights = []
            points = generate_points_on_sphere(radius=1,number_points=2*number_lights)
            camera_direction = cam.get_looking_direction()
            cos_angle = np.sum(-points * camera_direction.T,axis=1)
            min_cos = np.cos(max_angle*np.pi/180)
            ind = np.where(cos_angle >= min_cos)[0]
            directions = -points[ind,:]
            for i,direction in enumerate(directions) :
                l = Light(id=str(k)+"_"+str(i),type="sun", direction=direction, color=colors[k], strength=strengths[k])
                cam_lights.append(l)
            self.lights.append(cam_lights)
        self.fixed_light = False

class Object():

    def __init__(self):
        self.type = None
        self.path = None
        self.texture_path = None
        self.subdivisions = None
        self.number_faces = None
        self.radius = None
        self.location = None
        self.rotation = None
        self.scale = None
        self.color = None
        self.ior = None

    def from_path(self,path,texture_path="",location=[],rotation=[],scale=[]):
        self.type = "path"
        self.path = path
        self.texture_path = texture_path
        self.rotation = rotation
        self.location = location
        self.scale = scale

    def as_cube(self,location,scale,rotation=[]):
        self.type = "cube"
        self.rotation = rotation
        self.location = location
        self.scale = scale

    def as_sphere(self,location,radius,subdivisions=4):
        self.type = "sphere"
        self.radius = radius
        self.location = location
        self.subdivisions = subdivisions

    def as_polyhedron(self,location,scale,number_faces,rotation=[]):
        self.type = "polyhedron"
        self.rotation = rotation
        self.location = location
        self.scale = scale
        self.number_faces = number_faces

    def add_refraction(self,color,ior):
        self.color = color
        self.ior = ior

class Scene():

    def __init__(self,cameras,lights,object,medium,output_path):

        assert(object.type=="path" or object.type=="sphere")

        self.lights = lights
        self.cameras = cameras
        self.object = object
        self.medium = medium
        self.output_path = output_path
        self.render_without = False
        self.render_without_obj_mask = False
        self.render_with = False
        self.render_with_obj_mask = False
        self.render_with_medium_mask = False
        self.save_scene = True
        self.save_lights = False
        self.stereo_photometry = False

    def render_with_medium(self,state=True):
        self.render_with = state
        self.render_with_obj_mask = state
        self.render_with_medium_mask = state

    def render_without_medium(self,state):
        self.render_without = state
        self.render_without_obj_mask = state

    def save_lights_params(self,state):
        self.save_lights = state

    def stereo_photometry_rendering(self,state):
        self.stereo_photometry = state



