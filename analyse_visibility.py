import os.path

import numpy as np
from matplotlib import pyplot as plt


def decode_obj_file(file_path):
    vertices = []
    normals = []
    faces = []
    colors = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            tokens = line.split()
            prefix = tokens[0]

            if prefix == 'v':
                # Vertex position
                x, y, z = map(float, tokens[1:4])
                vertices.append((x, y, z))

                if len(tokens) > 4:
                    # Vertex color (optional)
                    r, g, b = map(float, tokens[4:7])
                    colors.append((r, g, b))

            elif prefix == 'f':
                face = []
                for token in tokens[1:]:
                    vertex_info = token.split('/')
                    vertex_index = int(vertex_info[0]) - 1
                    face.append(vertex_index)
                faces.append(face)

    return np.array(vertices), np.array(faces), np.array(colors)

def write_obj_file(file_path, vertices, faces, colors=None):
    with open(file_path, 'w') as file:
        for vertex,color in zip(vertices,colors):
            file.write(f'v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}')
            file.write(f' {color[0]:.6f} {color[1]:.6f} {color[2]:.6f}')
            file.write('\n')

        for face in faces:
            face_data = [f'{v+1}' for v in face]
            file.write(f'f {" ".join(face_data)}\n')

if __name__ == '__main__':

    folders = ["bearPNG","buddhaPNG","cowPNG","pot2PNG","readingPNG"]
    plt.figure()



    for k,name in enumerate(folders) :

        obj_file_path = 'D:/PhD/Dropbox/CVPR_2024/check_normals_and_visibility/'+name+'/GT/mesh_visibility.obj'
        vertices, faces, colors = decode_obj_file(obj_file_path)
        colors_2 = np.round(20 * colors[:, 0]).astype(int)

        colors_seuil = np.where(colors[:,0] < 1 / 20.0,
                                0,
                                0.25)
        colors_seuil = np.where(colors[:,0] < 6 / 20.0,
                                colors_seuil,
                                0.5)
        colors_seuil = np.where(colors[:,0] < 11 / 20.0,
                                colors_seuil,
                                0.75)
        colors_seuil = np.where(colors[:,0] < 16/20.0,
                              colors_seuil,
                              1)




        colors_seuils = plt.cm.jet(colors_seuil)[:,:3]

        print(vertices.shape)
        print(colors_seuil.shape)
        print(colors_seuils.shape)

        if not os.path.exists('D:/PhD/Dropbox/CVPR_2024/check_normals_and_visibility/'+name+'/GT_visibility/'):
            os.mkdir('D:/PhD/Dropbox/CVPR_2024/check_normals_and_visibility/'+name+'/GT_visibility/')

        write_obj_file('D:/PhD/Dropbox/CVPR_2024/check_normals_and_visibility/'+name+'/GT_visibility/mesh_visibility_seuil.obj',vertices,faces,colors_seuils)

        plt.subplot(2,3,k+1)
        plt.hist(colors_2,bins=len(set(colors_2))-1)
        plt.title(name)
    plt.show()
