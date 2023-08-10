import os.path

import numpy as np
import cv2
import os


if __name__ == '__main__':

    folder = "/home/robin/Desktop/PhD/Data//Data_ortho_normal_full_2/"
    RR = np.load(folder + "R.npy")
    TT = np.load(folder + "T.npy")
    K = np.load(folder + "K.npy")

    interface_old = np.load(folder + "interface.npz")
    interface = {}
    for e in interface_old.files :
        interface.update({e:interface_old[e].T})

    if not os.path.exists(folder+"plan_id/"):
        os.mkdir(folder+"plan_id/")

    for view in range(0,RR.shape[2]):

        R = RR[:,:,view].reshape(3,3)
        T = TT[:,[view]]

        cam_direction = (R.T @ (np.array([0,0,1]).reshape(3,1))).reshape(3)

        image_plan_id = 255*np.ones((1080,1920))
        for k in range(len(interface["faces"])):
            ff = interface["faces"][k].astype(int)
            vv = interface["vertices"][ff,:].T
            nn = interface["normals"][k]
            nn = nn//np.linalg.norm(nn)
            if np.dot(nn,-cam_direction) > 1e-8 :
                vv_cam = R @ vv + T
                vv_cam[2,:] = 1.0
                vv_pixel = np.round(K @ vv_cam).astype(int)
                triangle_cnt = np.array([vv_pixel[:2, 0], vv_pixel[:2, 1], vv_pixel[:2, 2]])
                triangle_cnt = triangle_cnt.reshape((-1, 1, 2)).astype(np.int32)
                coco = (k, k, k)
                cv2.drawContours(image_plan_id, [triangle_cnt], 0, coco, -1)
        cv2.imwrite(folder+"plan_id/"+(3-len(str(view)))*"0"+str(view)+".png",image_plan_id)







