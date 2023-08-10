import pickle
import numpy as np

def vector_to_array(v):
    return np.array([v.x,v.y,v.z])

class vertice():
    def __init__(self,position,faces,normal,id):
        self.position = position
        self.normal = normal
        self.id = id
        self.faces = faces
        self.faces_id = []
        self.respect_distance = None

class face():
    def __init__(self,l_vertices,id_f):
        self.vertices = l_vertices
        self.vertices_id = None
        self.id = id_f
        self.area = None
        self.normal = None
        self.center = None
        self.neighboors = None

    def get_vert_ind(self):
        v_ind = []
        for v in self.vertices :
            v_ind.append(v.id)
        return v_ind

class object_3D():

    def __init__(self):

        self.vertices = None
        self.faces = None

    def init_from_blender(self,matrix_world,vertices,faces,doublon=True):

        self.vertices = [vertice(None,[],None,i) for i in range(len(vertices))]
        self.faces = []
        self.vertices_as_tab = np.zeros((len(vertices),3))
        for k in range(0, len(faces)):
            face_vertices = faces[k].vertices

            l_v = []
            l_v_id = []
            f = face(None,k)
            for ind_v in face_vertices :
                l_v.append(self.vertices[ind_v])
                l_v_id.append(ind_v)
                if f not in self.vertices[ind_v].faces :
                    self.vertices[ind_v].faces.append(f)
                    self.vertices[ind_v].faces_id.append(k)
                vect = vector_to_array(matrix_world @ vertices[ind_v].co)
                self.vertices[ind_v].position = vect
                self.vertices_as_tab[ind_v,:] = vect
                normal = (vector_to_array(vertices[ind_v].normal)).reshape(3,1)
                new_normal = np.dot(np.matrix(matrix_world)[:3,:3],normal)
                self.vertices[ind_v].normal = np.array([new_normal[0,0],new_normal[1,0],new_normal[2,0]])
            f.vertices = l_v
            f.vertices_id = l_v_id
            self.faces.append(f)

        if doublon:
            print("check doublon")
            self.check_doublon()

    def compute_face_center_normal(self):
        for face in self.faces :
            position = []
            for vv in face.vertices:
                position.append(vv.position)
            face.normal = np.cross(position[1]-position[0],position[2]-position[0])
            face.center = (position[0]+position[1]+position[2])/3

    def save_data_as_dict(self,output_file):
        tab_vertices = np.zeros((len(self.vertices),3))
        tab_faces = np.zeros((len(self.faces),3))
        tab_normals = np.zeros((len(self.faces),3))
        tab_centers = np.zeros((len(self.faces),3))

        for k,vv in enumerate(self.vertices):
            tab_vertices[k,:] = vv.position
        for k, ff in enumerate(self.faces):
            tab_faces[k, 0] = ff.vertices[0].id
            tab_faces[k, 1] = ff.vertices[1].id
            tab_faces[k, 2] = ff.vertices[2].id
            tab_normals[k,:] = ff.normal
            tab_centers[k,:] = ff.center


        f=open(output_file,'wb')
        data = {"vertices":tab_vertices,"faces":tab_faces,"centers":tab_centers,"normals":tab_normals}
        data2 = {"vertices": tab_vertices.T, "faces": tab_faces.T, "centers": tab_centers.T, "normals": tab_normals.T}
        pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
        np.savez(output_file[:-3] + "npz", **data2)

    def check_doublon(self):

        n = len(self.vertices)
        l_doublon = []
        l_j = []
        for i in range(n):
            v1 = self.vertices[i]
            local = []
            for j in range(i+1,n):
                if j not in l_j:
                    v2 = self.vertices[j]
                    if np.linalg.norm(v1.position-v2.position) < 1e-15 :
                        l_j.append(j)
                        local.append(j)
            l_doublon.append([i,local])


        for doublon in l_doublon :
            v_keep = self.vertices[doublon[0]]
            for copy in doublon[1]:
                v_copy = self.vertices[copy]
                for face in v_copy.faces :
                    if face not in v_keep.faces :
                        v_keep.faces.append(face)
                    face.vertices.remove(v_copy)
                    face.vertices.append(v_keep)


