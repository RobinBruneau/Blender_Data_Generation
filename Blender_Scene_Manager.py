from _tools.data_structure import Scene,CameraManager,LightManager,Object
import numpy as np

def generate_blender_scene():

    # PATH
    object_path = "D:/PhD/Dropbox/Data/Data/models/Graphosoma/Graphosoma.obj"
    object_texture_path = "D:/PhD/Dropbox/Data/Data/models/Graphosoma/Graphosoma.png"
    medium_path = ""
    output_path = ("D:/PhD/Projects/Playing_with_NeuS/data/Boudha_marble_1920_1080_MVPS/")

    # GENERATE CAMERAS
    #    - sphere_cameras (cameras on a sphere of choosen radius
    #    - ring_cameras (cameras on a z-height radius ring
    #    - single_camera (give your location and rotation/look_at)
    #    - multi_cameras (gives your locations and details)
    cm = CameraManager()
    #size = (320,180)
    size = (1920, 1080)
    # ortho = 1.7
    # perspec = 50
    # cm.sphere_cameras(radius=5,number_cameras=20,lens=4,type="orthographic")
    R = np.load("D:/PhD/Projects/Playing_with_NeuS/data/Graphosoma_MVPS/R.npy")
    T = np.load("D:/PhD/Projects/Playing_with_NeuS/data/Graphosoma_MVPS/T.npy")
    C = []
    for k in range(R.shape[2]):
        C.append((-R[:,:,k].reshape(3,3).T @ T[:,[k]]).reshape(3))
    #cm.from_camera_RT(C,lens=50)
    cm.size = size
    cm.depth_bit = '16'
    cm.sphere_cameras(radius=11.5, number_cameras=20, lens=50, type="perspective",size=size)
    #cm.single_camera(location=[-10,0,0],rotation=[0,0,0],lens=1.7,is_looking_at=True,looking_at=[0.0,0.0,0.0],type="orthographic")
    #cm.multi_cameras(locations=[[3,0,0],[0,3,0],[0,0,3]],rotations=[[0,0,0],[0,0,0],[0,0,0]],lens=50,is_looking_at=True,looking_at=[[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]],type="perspective")
    #cm.cameras = [cm.cameras[0]]

    # GENERATE LIGHTS
    #    - fixed_ambiant (an ambiant light for all cameras)
    #    - fixes_directional (a set of directional lights for all cameras)
    #    - semi_sphere_directionnal_per_camera (a set of directional lights for each camera)
    lm = LightManager()
    lm.add_ambiant(0.5)
    #lm.fixed_ambiant(color=(1.0,1.0,1.0,1.0),strength=0.5)
    #lm.semi_sphere_area_per_camera(radius=10,cameras=cm.cameras,number_lights=10,colors=[(1.0,1.0,1.0) for i in range(len(cm.cameras))],strengths=[1500 for i in range(len(cm.cameras))],max_angle=67.5)

    lights_data = np.load("D:/PhD/Projects/Playing_with_NeuS/data/normal_sdmunips/lights_canonic_3.npz")
    lights = []
    for k in range(R.shape[2]):
        lights.append(lights_data["cam_{}".format(k)])
    #lm.from_ligth_npz(cameras=cm.cameras,lights=lights, colors=[(1.0, 1.0, 1.0) for i in range(len(cm.cameras))],strengths=[np.pi for i in range(len(cm.cameras))])
    lm.semi_sphere_directionnal_per_camera(cameras=cm.cameras, number_lights=3, colors=[(1.0, 1.0, 1.0) for i in range(len(cm.cameras))],strengths=[2.5 for i in range(len(cm.cameras))], max_angle=67.5)
    #lm.fixed_directionnals(directions=[np.array([1.0,0.0,0.0]).reshape(3,1),np.array([0.0,1.0,0.0]).reshape(3,1),np.array([0.0,0.0,1.0]).reshape(3,1)],colors = [(1.0,1.0,1.0) for i in range(3)],strengths = [np.pi,np.pi,np.pi])

    # OBJECT
    #   - from_path (loading a .obj mesh)
    object = Object()
    #object.as_sphere(location=[0.0,0.0,0.0],radius=0.25,subdivisions=5)
    object.from_path(path=object_path,texture_path=object_texture_path,scale=[1.0,1.0,1.0])
    #object.specific_material("001")

    # MEDIUM
    #    - as_cube (a cube shape)
    #    - as_sphere (a sphere shape)
    #    - as_polyhedron (a polyhedron with a chosen number of faces)
    #    - from_path (loading a .obj mesh)
    #  + add_refraction (add refractive BSDF : color + ior)
    medium = Object()
    medium.as_cube(location=[0.0,0.0,0.0],rotation=[],scale=[1.0,1.0,1.0])
    medium.add_refraction(color=(1.0,1.0,1.0,1.0),ior=1.56)

    # SCENE
    #   - render_with_medium ((de)activate the rendering with the refractive medium)
    #   - render_without_medium ((de)activate the rendering without the refractive medium)
    scene = Scene(cm,lm,object,medium,output_path)
    scene.render_with_medium(state=False)
    scene.render_without_medium(state=True)
    assert(scene.render_with ^ scene.render_without)
    scene.save_lights_params(state=True)
    scene.stereo_photometry_rendering(state=False)

    return scene
