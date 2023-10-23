import os.path
from _tools.data_structure import Scene,CameraManager,LightManager,Object
import numpy as np

def generate_blender_scene(folder,object_path):

    # PATH
    output_path = (folder)

    data_cam = np.load(folder+"cameras_unpacked.npz")
    K = data_cam["K"]
    R = data_cam["R"]
    T = data_cam["T"]

    C = []
    look_at = []
    for k in range(R.shape[0]):
        C.append((-R[k, :, :].reshape(3, 3).T @ T[[k],:]).reshape(3))
        look_at.append((R[k, :, :].reshape(3, 3).T @ (np.array([[0],[0],[1]])-T[[k],:])).reshape(3))

    sx = int(np.round(K[0,0,2]*2))
    sy = int(np.round(K[0,1,2]*2))
    lens = K[0,0,0]*36/sx

    # GENERATE CAMERAS
    cm = CameraManager()
    size = (sx,sy)
    cm.from_camera_RT(C,look_at,lens=lens)
    cm.size = size
    cm.depth_bit = '16'


    # OBJECT
    object = Object()
    object.from_path(path=object_path,scale=[1.0,1.0,1.0])

    # SCENE
    scene = Scene(cm,object,output_path)
    scene.render_normal_maps(state=True)

    return scene
