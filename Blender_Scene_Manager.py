from _tools.data_structure import Scene,CameraManager,LightManager,Object

def generate_blender_scene():

    # PATH
    object_path = "D:/INPUT/Graphosoma/Graphosoma.obj"
    object_texture_path = "D:/INPUT/Graphosoma/Graphosoma.png"
    medium_path = ""
    output_path = "D:/PhD/Projects/Blender_Data_Generation/OUTPUT/test_1/"

    # GENERATE CAMERAS
    #    - sphere_cameras (cameras on a sphere of choosen radius
    #    - ring_cameras (cameras on a z-height radius ring
    #    - single_camera (give your location and rotation/look_at)
    #    - multi_cameras (gives your locations and details)
    cm = CameraManager()
    cm.sphere_cameras(radius=3,number_cameras=16,lens=50)
    #cm.cameras = [cm.cameras[0]]

    # GENERATE LIGHTS
    #    - fixed_ambiant (an ambiant light for all cameras)
    #    - fixes_directional (a set of directional lights for all cameras)
    #    - semi_sphere_directionnal_per_camera (a set of directional lights for each camera)
    lm = LightManager()
    lm.fixed_ambiant(color=(1.0,1.0,1.0,1.0),strength=1)
    #lm.semi_sphere_directionnal_per_camera(cm.cameras,20,colors=[(0.0,0.0,0.0)],strengths=[1])

    # OBJECT
    #   - from_path (loading a .obj mesh)
    object = Object()
    object.from_path(path=object_path,texture_path=object_texture_path,scale=[0.07,0.07,0.07])

    # MEDIUM
    #    - as_cube (a cube shape)
    #    - as_sphere (a sphere shape)
    #    - as_polyhedron (a polyhedron with a chosen number of faces)
    #    - from_path (loading a .obj mesh)
    #  + add_refraction (add refractive BSDF : color + ior)
    medium = Object()
    medium.as_cube(location=[0.0,0.0,0.0],rotation=[],scale=1)
    medium.add_refraction(color=(1.0,1.0,1.0,1.0),ior=1.56)

    # SCENE
    #   - render_with_medium ((de)activate the rendering with the refractive medium)
    #   - render_without_medium ((de)activate the rendering without the refractive medium)
    scene = Scene(cm,lm,object,medium,output_path)
    scene.render_with_medium(state=False)
    scene.render_without_medium(state=False)

    return scene