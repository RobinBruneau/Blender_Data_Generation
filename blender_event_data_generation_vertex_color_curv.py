import glob
import argparse
import bpy
import numpy as np
import os
import sys
from pathlib import Path
sys.path.append(os.path.dirname(__file__))
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from _tools.object_3D import object_3D
from Blender_Scene_Manager import generate_blender_scene
import random
import easy_keys


class ModalTimerOperator():
    """Operator which runs itself from a timer"""
    bl_idname = "wm.modal_timer_operator"
    bl_label = "Modal Timer Operator"


    def __init__(self,folder,mesh,output):

        self.scene_parameters = generate_blender_scene(folder,mesh)
        self.global_light = None
        self.global_light_color = None
        self.global_light_intensity = None
        self._scene_data = None
        self.first_F1 = True
        self.first_F5 = True
        self.first_F6 = True
        self.first_F7 = True
        self.not_finished = True
        self.not_f7 = True
        self.not_f8 = True
        self.output = output

        self.cryptomatte_node = None
        self.viewer_node = None

        self.l_render = []
        self.output_render = []
        self.l_object = []
        self.ind_render = 0
        self.ind_cam_k = 0
        self.ind_object = 0
        self.nb_cam = None
        self.empty_centered_point = None

    def generate_plans(self):

        material_black = bpy.data.materials.new(name="black")
        material_black.use_nodes = True
        bsdf = material_black.node_tree.nodes.get('Principled BSDF')
        bsdf.inputs[0].default_value = (0, 0, 0, 1)

        bpy.ops.mesh.primitive_plane_add(location=(0.701, 0, 0), size = 1.4,rotation=(0,np.pi/2,0))
        p1 = bpy.context.active_object
        p1.hide_render=True
        p1.active_material = material_black
        bpy.ops.mesh.primitive_plane_add(location=(-0.701, 0, 0), size = 1.4, rotation=(0, np.pi / 2, 0))
        p2 = bpy.context.active_object
        p2.hide_render = True
        p2.active_material = material_black
        bpy.ops.mesh.primitive_plane_add(location=(0, 0, 0.701), size = 1.4)
        p3 = bpy.context.active_object
        p3.hide_render = True
        p3.active_material = material_black
        bpy.ops.mesh.primitive_plane_add(location=(0, 0, -0.701), size = 1.4)
        p4 = bpy.context.active_object
        p4.hide_render = True
        p4.active_material = material_black
        bpy.ops.mesh.primitive_plane_add(location=(0,0.701, 0), size = 1.4, rotation=(np.pi / 2, 0, 0))
        p5 = bpy.context.active_object
        p5.hide_render = True
        p5.active_material = material_black
        bpy.ops.mesh.primitive_plane_add(location=(0,-0.701, 0), size = 1.4, rotation=(np.pi / 2, 0, 0))
        p6 = bpy.context.active_object
        p6.hide_render = True
        p6.active_material = material_black

        return [p1,p2,p3,p4,p5,p6]





    def clean_scene(self):
        # Clean the scene
        for c in bpy.context.scene.collection.children:
            bpy.context.scene.collection.children.unlink(c)

    def generate_fixed_scene(self):

        self.clean_scene()
        camera_collection = bpy.data.collections.new("Cameras")
        light_collection = bpy.data.collections.new("Lights")
        objects_collection = bpy.data.collections.new("Object_Medium")
        look_at_collection = bpy.data.collections.new("Look_at")
        bpy.context.scene.collection.children.link(camera_collection)
        bpy.context.scene.collection.children.link(light_collection)
        bpy.context.scene.collection.children.link(objects_collection)
        bpy.context.scene.collection.children.link(look_at_collection)

        _object = self.generate_object()
        all_cams = self.generate_cameras()
        self.apply_rendering_params()

        bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (0.0,0.0,0.0,1.0)
        bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = 0


        _data = {"object": _object, "all_cams": all_cams}

        return _data

    def global_white_light(self,state):
        bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (1.0,1.0,1.0,1.0)
        bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = 1

    def generate_cameras(self):

        all_cams = []
        cameras_parameters = self.scene_parameters.cameras.cameras
        for cam_parameters in cameras_parameters :
            cam_data = bpy.data.cameras.new(cam_parameters.id)
            cam = bpy.data.objects.new(cam_parameters.id, cam_data)
            cam.location = cam_parameters.location
            if cam_parameters.is_looking_at :
                cam.rotation_euler = [0.0, 0.0, 0.0]
                constraint = cam.constraints.new(type='TRACK_TO')
                bpy.ops.object.empty_add(type='CIRCLE', align='WORLD', location=cam_parameters.looking_at, scale=(0, 0, 0))
                look_at = bpy.context.selected_objects[0]
                bpy.context.scene.collection.objects.unlink(look_at)
                bpy.data.collections["Look_at"].objects.link(look_at)
                constraint.target = look_at
            else :
                cam.rotation_euler = cam_parameters.rotation
            if self.scene_parameters.cameras.type == "perspective" :
                cam.data.lens = cam_parameters.lens
            else :
                cam.data.type = "ORTHO"
                cam.data.ortho_scale = cam_parameters.lens

            cam.data.shift_x = cam_parameters.shift[0]
            cam.data.shift_y = cam_parameters.shift[1]

            cam.data.clip_end = 100000
            bpy.data.collections["Cameras"].objects.link(cam)
            all_cams.append(cam)
        return all_cams

    def generate_materials(self):

        # Create Object Material based on known Texture
        material_object_texture = bpy.data.materials.new(name="object_texture")
        material_object_texture.use_nodes = True
        material_object_texture.node_tree.nodes.remove(material_object_texture.node_tree.nodes.get('Principled BSDF'))
        NodeBSDF = material_object_texture.node_tree.nodes.new('ShaderNodeBsdfDiffuse')
        NodeBSDF.inputs["Roughness"].default_value = 0.0
        if self.scene_parameters.object.type == "path" :
            Node_TEX = material_object_texture.node_tree.nodes.new('ShaderNodeTexImage')
            Node_TEX.image = bpy.data.images.load(self.scene_parameters.object.texture_path)
            if self.scene_parameters.stereo_photometry :
                Node_TEX.image.colorspace_settings.name = 'Raw'
            Node_output = material_object_texture.node_tree.nodes.get('Material Output')
            links = material_object_texture.node_tree.links
            links.new(Node_TEX.outputs["Color"], NodeBSDF.inputs["Color"])
            links.new(NodeBSDF.outputs["BSDF"], Node_output.inputs["Surface"])
        else :
            Node_output = material_object_texture.node_tree.nodes.get('Material Output')
            NodeBSDF.inputs["Color"].default_value = (1.0,1.0,1.0,1.0)
            links = material_object_texture.node_tree.links
            links.new(NodeBSDF.outputs["BSDF"], Node_output.inputs["Surface"])

        # Create Refractive medium Material
        material_refractive_medium = bpy.data.materials.new(name="mat_refractive")
        material_refractive_medium.use_nodes = True
        material_refractive_medium.node_tree.nodes.remove(
            material_refractive_medium.node_tree.nodes.get('Principled BSDF'))
        material_output = material_refractive_medium.node_tree.nodes.get('Material Output')
        #NodeBSDF = material_refractive_medium.node_tree.nodes.new('ShaderNodeBsdfGlass')
        NodeBSDF = material_refractive_medium.node_tree.nodes.new('ShaderNodeBsdfRefraction')
        NodeBSDF.inputs["Color"].default_value = self.scene_parameters.medium.color
        NodeBSDF.inputs["Roughness"].default_value = 0.0  # Make the glass fully transparent
        NodeBSDF.inputs["IOR"].default_value = self.scene_parameters.medium.ior  # IOR
        NodeBSDF.distribution = "GGX"
        material_refractive_medium.node_tree.links.new(material_output.inputs[0], NodeBSDF.outputs[0])

        prefs = bpy.context.preferences
        filepaths = prefs.filepaths
        asset_libraries = filepaths.asset_libraries

        for asset_library in asset_libraries:
            library_name = asset_library.name
            library_path = Path(asset_library.path)
            blend_files = [fp for fp in library_path.glob("**/*.blend") if fp.is_file()]
            print(f"Checking the content of library '{library_name}' :")
            for blend_file in blend_files:
                with bpy.data.libraries.load(str(blend_file), assets_only=True) as (data_from, data_to):
                    data_to.materials = data_from.materials

        all_materials = {"object": material_object_texture,
                         "refractive_medium": material_refractive_medium}

        for m in bpy.data.materials:
            all_materials.update({m.name:m})



        return all_materials

    def generate_object(self):

        object_parameters = self.scene_parameters.object
        if object_parameters.type == "path" :

            if object_parameters.path[-1] == "y" :
                bpy.ops.import_mesh.ply(filepath=object_parameters.path)
            else :
                bpy.ops.wm.obj_import(filepath=object_parameters.path)

            _object = bpy.context.selected_objects[0]
            _object.name = "Object"

            if len(object_parameters.location) != 0 :
                _object.location.x = object_parameters.location[0]
                _object.location.y = object_parameters.location[1]
                _object.location.z = object_parameters.location[2]

            if len(object_parameters.scale) != 0 :
                _object.scale.x = object_parameters.scale[0]
                _object.scale.y = object_parameters.scale[1]
                _object.scale.z = object_parameters.scale[2]


            _object.rotation_euler.x = 0
            _object.rotation_euler.y = 0
            _object.rotation_euler.z = 0

            bpy.context.view_layer.update()
            bpy.context.scene.collection.objects.unlink(_object)
            bpy.data.collections["Object_Medium"].objects.link(_object)

        # Create Object Material based on known Texture
        material_object_texture = bpy.data.materials.new(name="object_texture")
        material_object_texture.use_nodes = True
        material_object_texture.node_tree.nodes.remove(
            material_object_texture.node_tree.nodes.get('Principled BSDF'))
        NodeBSDF = material_object_texture.node_tree.nodes.new('ShaderNodeEmission')
        Node_color = material_object_texture.node_tree.nodes.new('ShaderNodeVertexColor')
        Node_color.layer_name = "Color"

        Node_output = material_object_texture.node_tree.nodes.get('Material Output')
        links = material_object_texture.node_tree.links
        links.new(Node_color.outputs["Color"], NodeBSDF.inputs["Color"])
        links.new(NodeBSDF.outputs["Emission"], Node_output.inputs["Surface"])

        _object.active_material = material_object_texture

        return _object

    def generate_refractive_medium(self):

        def generate_sphere(medium_parameters):
            bpy.ops.mesh.primitive_ico_sphere_add(radius=medium_parameters.radius, location=medium_parameters.location, subdivisions=medium_parameters.subdivisions)
            refractive_medium = bpy.context.active_object
            refractive_medium.name = "Refractive Medium"
            bpy.context.view_layer.update()
            bpy.ops.object.shade_smooth()
            bpy.context.view_layer.update()
            return refractive_medium

        def generate_cube(medium_parameters):
            bpy.ops.mesh.primitive_cube_add(size=1, location=medium_parameters.location)
            refractive_medium = bpy.context.active_object
            refractive_medium.name = "Refractive Medium"

            if len(medium_parameters.scale) != 0 :
                refractive_medium.scale.x = medium_parameters.scale[0]
                refractive_medium.scale.y = medium_parameters.scale[1]
                refractive_medium.scale.z = medium_parameters.scale[2]

            if len(medium_parameters.rotation) != 0:
                refractive_medium.rotation_euler.x = medium_parameters.rotation[0]
                refractive_medium.rotation_euler.y = medium_parameters.rotation[1]
                refractive_medium.rotation_euler.z = medium_parameters.rotation[2]

            bpy.ops.object.modifier_add(type='TRIANGULATE')
            bpy.ops.object.modifier_apply(modifier="Triangulate")
            return refractive_medium

        def generate_polyhedron(medium_parameters):
            bpy.ops.mesh.primitive_solid_add(source=str(medium_parameters.number_faces), size=medium_parameters.scale,location=medium_parameters.location)
            refractive_medium = bpy.context.active_object
            refractive_medium.name = "Refractive Medium"

            if len(medium_parameters.rotation) != 0:
                refractive_medium.rotation_euler.x = medium_parameters.rotation[0]
                refractive_medium.rotation_euler.y = medium_parameters.rotation[1]
                refractive_medium.rotation_euler.z = medium_parameters.rotation[2]

            bpy.ops.object.modifier_add(type='TRIANGULATE')
            bpy.ops.object.modifier_apply(modifier="Triangulate")
            return refractive_medium

        def generate_obj(medium_parameters):
            bpy.ops.import_scene.obj(filepath=medium_parameters.path)
            refractive_medium = bpy.context.selected_objects[0]
            refractive_medium.name = "Refractive Medium"
            if len(medium_parameters.location) != 0:
                refractive_medium.location.x = medium_parameters.location[0]
                refractive_medium.location.y = medium_parameters.location[1]
                refractive_medium.location.z = medium_parameters.location[2]

            if len(medium_parameters.scale) != 0:
                refractive_medium.scale.x = medium_parameters.scale[0]
                refractive_medium.scale.y = medium_parameters.scale[1]
                refractive_medium.scale.z = medium_parameters.scale[2]

            if len(medium_parameters.rotation) != 0:
                refractive_medium.rotation_euler.x = medium_parameters.rotation[0]
                refractive_medium.rotation_euler.y = medium_parameters.rotation[1]
                refractive_medium.rotation_euler.z = medium_parameters.rotation[2]
            return refractive_medium

        medium_parameters = self.scene_parameters.medium
        if medium_parameters.type == "cube":
            refractive_medium = generate_cube(medium_parameters)
        elif medium_parameters.type == "sphere":
            refractive_medium = generate_sphere(medium_parameters)
        elif medium_parameters.type == "polyhedron":
            refractive_medium = generate_polyhedron(medium_parameters)
        else :
            refractive_medium = generate_obj(medium_parameters)
        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.scene.collection.objects.unlink(refractive_medium)
        bpy.data.collections["Object_Medium"].objects.link(refractive_medium)
        return refractive_medium

    def get_calibration_matrix_K_from_blender(self,cam):

        def get_sensor_size(sensor_fit, sensor_x, sensor_y):
            if sensor_fit == 'VERTICAL':
                return sensor_y
            return sensor_x

        def get_sensor_fit(sensor_fit, size_x, size_y):
            if sensor_fit == 'AUTO':
                if size_x >= size_y:
                    return 'HORIZONTAL'
                else:
                    return 'VERTICAL'
            return sensor_fit

        camd = cam.data
        if camd.type != 'PERSP':
            scene = bpy.context.scene
            scale = camd.ortho_scale
            rx = scene.render.resolution_x
            ry = scene.render.resolution_y
            return np.array([[rx / scale, 0, rx / 2], [0, rx / scale, ry / 2], [0, 0, 1]])

        scene = bpy.context.scene
        f_in_mm = camd.lens
        scale = scene.render.resolution_percentage / 100
        resolution_x_in_px = scale * scene.render.resolution_x
        resolution_y_in_px = scale * scene.render.resolution_y
        sensor_size_in_mm = get_sensor_size(camd.sensor_fit, camd.sensor_width, camd.sensor_height)
        sensor_fit = get_sensor_fit(
            camd.sensor_fit,
            scene.render.pixel_aspect_x * resolution_x_in_px,
            scene.render.pixel_aspect_y * resolution_y_in_px
        )
        pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
        if sensor_fit == 'HORIZONTAL':
            view_fac_in_px = resolution_x_in_px
        else:
            view_fac_in_px = pixel_aspect_ratio * resolution_y_in_px
        pixel_size_mm_per_px = sensor_size_in_mm / f_in_mm / view_fac_in_px
        s_u = 1 / pixel_size_mm_per_px
        s_v = 1 / pixel_size_mm_per_px / pixel_aspect_ratio

        # Parameters of intrinsic calibration matrix K
        u_0 = resolution_x_in_px / 2 - camd.shift_x * view_fac_in_px
        v_0 = resolution_y_in_px / 2 + camd.shift_y * view_fac_in_px / pixel_aspect_ratio
        skew = 0  # only use rectangular pixels

        K = np.array([[s_u, skew, u_0],
                      [0, s_v, v_0],
                      [0, 0, 1]])
        return K

    def get_3x4_RT_matrix_from_blender(self,obj):
        # bcam stands for blender camera
        R_bcam2cv = np.array(
            [[1, 0, 0],
             [0, -1, 0],
             [0, 0, -1]])

        # Transpose since the rotation is object rotation,
        # and we want coordinate rotation
        # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
        # T_world2bcam = -1*R_world2bcam * location
        #
        # Use matrix_world instead to account for all constraints
        location, rotation = obj.matrix_world.decompose()[0:2]
        R_world2bcam = rotation.to_matrix().transposed()

        # Convert camera location to translation vector used in coordinate changes
        # T_world2bcam = -1*R_world2bcam*cam.location
        # Use location from matrix_world to account for constraints:
        T_world2bcam = -1 * R_world2bcam @ location

        # Build the coordinate transform matrix from world to computer vision camera
        # NOTE: Use * instead of @ here for older versions of Blender
        R_world2cv = R_bcam2cv @ R_world2bcam
        T_world2cv = R_bcam2cv @ T_world2bcam

        return R_world2cv, T_world2cv

    def save_lights_params(self):
        lights_param = self.scene_parameters.lights

        if self.scene_parameters.save_lights :
            l_directions = []
            if lights_param.fixed_light :
                for light in lights_param.lights :
                    l_directions.append(light.direction)
                data = {"global":l_directions}
            else :
                data = {}

                for k,light_cam in enumerate(lights_param.lights) :
                    cam_light_directions = []
                    cam_light_position = []
                    for light in light_cam :
                        cam_light_directions.append(-light.direction)
                        cam_light_position.append(light.position)
                    data.update({"cam_{}".format(k):cam_light_directions})
                    data.update({"cam_pos_{}".format(k):cam_light_position})

            np.savez(self.scene_parameters.output_path+"/lights.npz",**data)

        print('\033[93m' + "SAVE LIGHTS DONE\n\n" + '\033[0m')



    def save_cam_params(self):

        _all_cameras = self._scene_data["all_cams"]
        K = self.get_calibration_matrix_K_from_blender(_all_cameras[0])
        _all_R = np.zeros((3, 3, len(_all_cameras)))
        _all_T = np.zeros((3, len(_all_cameras)))
        data = {}
        for k, cam_k in enumerate(_all_cameras):
            R, T = self.get_3x4_RT_matrix_from_blender(cam_k)
            P = K @ np.concatenate((R,T.reshape(3,1)),axis=1)
            P = np.concatenate((P,np.array([0,0,0,1]).reshape(1,-1)),axis=0)
            data.update({"world_mat_{}".format(k):P})
            _all_R[:, :, k] = R
            _all_T[:, k] = T

        np.savez(self.scene_parameters.output_path + "/cameras.npz", **data)
        np.save(self.scene_parameters.output_path + "/R.npy", _all_R)
        np.save(self.scene_parameters.output_path + "/T.npy", _all_T)
        np.save(self.scene_parameters.output_path + "/K.npy", K)
        print('\033[93m' + "\nSAVE R/T/K DONE\n" + '\033[0m')

    def save_interface_data(self):

        bpy.ops.object.select_all(action='DESELECT')
        _object = self._scene_data["object"]
        refractive_medium = self._scene_data["refractive_medium"]
        bpy.context.view_layer.objects.active = _object
        bpy.context.view_layer.objects.active = refractive_medium

        m = object_3D()
        vv = refractive_medium.data.vertices
        mw = refractive_medium.matrix_world
        faces = refractive_medium.data.polygons
        m.init_from_blender(mw, vv, faces, doublon=False)
        m.compute_face_center_normal()
        m.save_data_as_dict(self.scene_parameters.output_path + "/interface.pkl")


        print('\033[93m' + "SAVE INTERFACE DONE\n\n" + '\033[0m')

    def activate_film(self,state):
        bpy.context.scene.render.film_transparent = state
        bpy.context.scene.cycles.film_transparent_glass = state

    def compositing_cryptomatte(self):

        # switch on nodes and get reference
        bpy.context.scene.use_nodes = True
        tree = bpy.context.scene.node_tree

        # clear default nodes
        for node in tree.nodes:
            tree.nodes.remove(node)

        # create input image node
        image_node = tree.nodes.new(type='CompositorNodeRLayers')
        image_node.location = 0, 0

        # create output node
        comp_node = tree.nodes.new('CompositorNodeComposite')
        comp_node.location = 400, 0

        # create output node
        viewer_node = tree.nodes.new('CompositorNodeViewer')
        viewer_node.location = 600, 0

        cryptomatte_node = tree.nodes.new('CompositorNodeCryptomatteV2')
        cryptomatte_node.location = -400, 400
        cryptomatte_node.inputs[0].default_value = (1, 1, 1, 1)

        # link nodes
        links = tree.links
        link = links.new(image_node.outputs["Image"], comp_node.inputs["Image"])
        link = links.new(image_node.outputs["Image"], cryptomatte_node.inputs["Image"])
        link = links.new(cryptomatte_node.outputs["Matte"], viewer_node.inputs["Image"])

        return cryptomatte_node,viewer_node


    def compositing_real(self,add_mask=True,add_normal=True):
        # switch on nodes and get reference
        bpy.context.scene.use_nodes = True
        tree = bpy.context.scene.node_tree

        # clear default nodes
        for node in tree.nodes:
            tree.nodes.remove(node)

        # create input image node
        image_node = tree.nodes.new(type='CompositorNodeRLayers')
        image_node.location = 0, 0

        # create output node
        comp_node = tree.nodes.new('CompositorNodeComposite')
        comp_node.location = 400, 0

        output_node = tree.nodes.new('CompositorNodeOutputFile')
        output_node.name = "mask"
        output_node.location = 400, 400
        output_node.format.file_format = "JPEG"

        normal_output_node = tree.nodes.new('CompositorNodeOutputFile')
        normal_output_node.name = "normal"
        normal_output_node.location = 800, 400
        normal_output_node.format.file_format = "JPEG"

        curve_rgb_node = tree.nodes.new(type="CompositorNodeCurveRGB")
        red = curve_rgb_node.mapping.curves[0]
        loc = red.points[0].location
        loc[1] = 0.5
        green = curve_rgb_node.mapping.curves[1]
        loc = green.points[0].location
        loc[1] = 0.5
        blue = curve_rgb_node.mapping.curves[2]
        loc = blue.points[0].location
        loc[1] = 0.5

        # link nodes
        links = tree.links
        link = links.new(image_node.outputs["Image"], comp_node.inputs[0])
        if add_mask :
            link = links.new(image_node.outputs["IndexOB"], output_node.inputs[0])
        if add_normal :
            link = links.new(image_node.outputs["Normal"], curve_rgb_node.inputs["Image"])
            link = links.new(curve_rgb_node.outputs["Image"], normal_output_node.inputs[0])

        return output_node,normal_output_node

    def setup_real(self):
        bpy.context.scene.cycles.samples = 256
        _object = self._scene_data["object"]
        _all_cameras = self._scene_data["all_cams"]

    def apply_rendering_params(self,):

        bpy.context.scene.render.resolution_x = int(self.scene_parameters.cameras.size[0])
        bpy.context.scene.render.resolution_y = int(self.scene_parameters.cameras.size[1])
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.scene.render.image_settings.color_depth = self.scene_parameters.cameras.depth_bit
        bpy.context.scene.render.image_settings.compression = 0
        bpy.context.scene.view_settings.view_transform = 'Standard'
        bpy.context.scene.render.engine = 'BLENDER_EEVEE'
        bpy.context.scene.view_layers["ViewLayer"].use_pass_transmission_direct = True
        bpy.context.scene.view_layers["ViewLayer"].use_pass_object_index = True
        bpy.context.scene.view_layers["ViewLayer"].use_pass_object_index = True
        bpy.context.scene.view_layers["ViewLayer"].use_pass_cryptomatte_object = True
        bpy.context.scene.view_layers["ViewLayer"].use_pass_cryptomatte_asset = True
        bpy.context.scene.view_layers["ViewLayer"].use_pass_cryptomatte_material = True
        bpy.context.scene.view_layers["ViewLayer"].use_pass_cryptomatte_material = True
        bpy.context.scene.view_layers["ViewLayer"].use_pass_normal = True
        #bpy.context.scene.display_settings.display_device = 'None'
        bpy.context.scene.render.pixel_aspect_x = 1
        bpy.context.scene.render.pixel_aspect_y = self.scene_parameters.ratio_f

    def render_object_color(self):

        bpy.context.scene.eevee.taa_render_samples = 512
        _object = self._scene_data["object"]
        _all_cameras = self._scene_data["all_cams"]

        for cam_num, cam_k in enumerate(_all_cameras):
            bpy.context.scene.camera = cam_k
            bpy.context.scene.render.filepath = self.scene_parameters.output_path +"/"+self.output+"/"+ cam_k.name+'.png'
            bpy.ops.render.render(write_still=1)

    def create_output_folders(self):
        if not os.path.exists(self.scene_parameters.output_path):
            os.mkdir(self.scene_parameters.output_path)
        if not os.path.exists(self.scene_parameters.output_path+self.output+"/"):
            os.mkdir(self.scene_parameters.output_path+self.output+"/")

    def rename(self):
        path_medium_mask = glob.glob(self.scene_parameters.output_path+"/normal/*cut*")
        for p in path_medium_mask :
            p = p.replace("\\","/")
            name = p.split("normal/")[-1].split("_cut_")[0]
            if os.path.exists(self.scene_parameters.output_path+"/normal/"+name+".png"):
                os.remove(self.scene_parameters.output_path+"/normal/"+name+".png")
            os.rename(p,self.scene_parameters.output_path+"/normal/"+name+".png")

        path_image_mask = glob.glob(self.scene_parameters.output_path + "/mask/*cut*")
        for p in path_image_mask:
            p = p.replace("\\", "/")
            name = p.split("mask/")[-1].split("_cut_")[0]
            if os.path.exists(self.scene_parameters.output_path+"/mask/"+name+".png"):
                os.remove(self.scene_parameters.output_path+"/mask/"+name+".png")
            os.rename(p, self.scene_parameters.output_path + "/mask/" + name + ".png")

        os.remove(self.scene_parameters.output_path+"/will_be_removed.png")


    def render(self):

        self.create_output_folders()
        self._scene_data = self.generate_fixed_scene()
        #bpy.ops.wm.save_as_mainfile(filepath=self.scene_parameters.output_path + "/scene.blend")
        self.render_object_color()
        #self.rename()
        bpy.ops.wm.quit_blender()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    known_args, unknown_args = parser.parse_known_args()
    print(unknown_args)

    if "--mesh" in unknown_args :
        mesh = unknown_args[unknown_args.index('--mesh')+1]
    if "--folder" in unknown_args:
        folder = unknown_args[unknown_args.index('--folder')+1]
    if "--output" in unknown_args:
        output = unknown_args[unknown_args.index('--output')+1]

    engine = ModalTimerOperator(folder,mesh,output)
    engine.render()

