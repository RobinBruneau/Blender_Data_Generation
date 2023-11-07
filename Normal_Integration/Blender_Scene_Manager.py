import os.path
from _tools.data_structure import Scene,CameraManager,LightManager,Object
import numpy as np

def generate_blender_scene(folder,object_path):

    # PATH
    output_path = (folder)

    data_cam = np.load(folder+"cameras_unpacked.npz")
    K = data_cam["K"].mean(axis=0)
    print(data_cam)
    R = data_cam["R"]
    R_euler = data_cam["R_euler"] * np.pi/180
    T = data_cam["T"]

    C = []

    ratio_f = K[1, 1] / K[0, 0]

    shift_x = (K[0, 2] - ((612 / 2) )) / 612
    shift_y = (K[1, 2]/ratio_f - ((512 / 2) )) / 512
    #shift_y = ((K[1, 2]  - ((512 / 2))) / 512 ) / ratio_f

    for k in range(R.shape[0]):
        C.append((-R[k, :, :].reshape(3, 3).T @ T[[k],:]).reshape(3))

    sx = int(np.round(K[0,2]*2))
    sy = int(np.round(K[1,2]*2))
    lens = float(K[0,0]*36/612)



    # GENERATE CAMERAS
    cm = CameraManager()
    size = (sx,sy)
    size = (612,512)
    cm.from_camera_RT(R_euler,C,shift=[shift_x,shift_y],lens=lens)
    cm.size = size
    cm.depth_bit = '16'


    # OBJECT
    object = Object()
    object.from_path(path=object_path,scale=[1.0,1.0,1.0])

    # SCENE
    scene = Scene(cm,object,output_path)
    scene.render_normal_maps(state=True)
    scene.change_ratio(ratio_f)

    return scene
