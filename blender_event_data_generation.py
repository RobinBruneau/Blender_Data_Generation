import json
import pickle

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


class ModalTimerOperator(bpy.types.Operator):
    """Operator which runs itself from a timer"""
    bl_idname = "wm.modal_timer_operator"
    bl_label = "Modal Timer Operator"


    def __init__(self):

        self.scene_parameters = generate_blender_scene()
        self.global_light = None
        self.global_light_color = None
        self.global_light_intensity = None
        self._scene_data = None
        self.first_F1 = True
        self.first_F5 = True
        self.first_F6 = True
        self.first_F7 = True
        self.not_f7 = True
        self.not_f8 = True

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

        if self.scene_parameters.render_with_obj_mask :
            self.l_render.append("masks")
            self.output_render.append(self.scene_parameters.output_path+"images_masks_with/")

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
        refractive_medium = self.generate_refractive_medium()
        all_cams = self.generate_cameras()
        all_lights = self.generate_lights()
        all_materials = self.generate_materials()
        self.apply_rendering_params()

        _data = {"object": _object, "refractive_medium": refractive_medium,
                 "all_cams": all_cams,"all_lights":all_lights, "all_materials": all_materials}

        return _data

    def global_white_light(self,state):

        if state :
            bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (1.0,1.0,1.0,1.0)
            bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = 10
        else :
            if self.global_light :
                bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = self.global_light_color
                bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = self.global_light_intensity
            else :
                bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = 0

    def generate_lights(self):

        bpy.ops.object.empty_add(type='CIRCLE', align='WORLD', location=(0, 0, 0), scale=(0, 0, 0))
        self.empty_centered_point = bpy.context.selected_objects[0]
        bpy.context.scene.collection.objects.unlink(self.empty_centered_point)
        bpy.data.collections["Look_at"].objects.link(self.empty_centered_point)

        # Disable world light
        bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = 0.0
        all_lights = []
        lights_parameters = self.scene_parameters.lights
        if lights_parameters.fixed_light :
            for k,light_param in enumerate(lights_parameters.lights):
                if light_param.type=="world" :
                    self.global_light = True
                    self.global_light_color = light_param.color
                    self.global_light_intensity = light_param.strength
                    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = light_param.color
                    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = light_param.strength
                else :
                    bpy.ops.object.light_add(type='SUN', align='WORLD', location=-light_param.direction, scale=(1, 1, 1))
                    light = bpy.context.selected_objects[0]
                    light.name = "directional_light_"+light_param.id
                    light.data.energy = light_param.strength
                    light.data.color = light_param.color
                    constraint = light.constraints.new(type='TRACK_TO')
                    constraint.target = self.empty_centered_point
                    bpy.context.scene.collection.objects.unlink(light)
                    bpy.data.collections["Lights"].objects.link(light)
                    all_lights.append(light)

        else :
            for k, light_group in enumerate(lights_parameters.lights):
                group = []
                for light_param in light_group :
                    bpy.ops.object.light_add(type='SUN', align='WORLD', location=-light_param.direction,
                                             scale=(1, 1, 1))
                    light = bpy.context.selected_objects[0]
                    light.name = "directional_light_"+light_param.id
                    light.data.energy = light_param.strength
                    light.data.color = light_param.color
                    light.hide_render = True
                    constraint = light.constraints.new(type='TRACK_TO')
                    constraint.target = self.empty_centered_point
                    bpy.context.scene.collection.objects.unlink(light)
                    bpy.data.collections["Lights"].objects.link(light)
                    group.append(light)
                all_lights.append(group)

        return all_lights

    def generate_cameras(self):

        all_cams = []
        cameras_parameters = self.scene_parameters.cameras.cameras
        for cam_parameters in cameras_parameters :
            cam_data = bpy.data.cameras.new('camera_{}'.format(cam_parameters.id))
            cam = bpy.data.objects.new('camera_{}'.format(cam_parameters.id), cam_data)
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
            cam.data.lens = cam_parameters.lens
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
        NodeBSDF = material_refractive_medium.node_tree.nodes.new('ShaderNodeBsdfGlass')
        NodeBSDF.inputs["Color"].default_value = self.scene_parameters.medium.color
        NodeBSDF.inputs["Roughness"].default_value = 0.0  # Make the glass fully transparent
        NodeBSDF.inputs["IOR"].default_value = self.scene_parameters.medium.ior  # IOR
        NodeBSDF.distribution = "MULTI_GGX"
        material_refractive_medium.node_tree.links.new(material_output.inputs[0], NodeBSDF.outputs[0])

        all_materials = {"object": material_object_texture,
                         "refractive_medium": material_refractive_medium}

        return all_materials

    def generate_object(self):

        object_parameters = self.scene_parameters.object
        if object_parameters.type == "path" :

            bpy.ops.import_scene.obj(filepath=object_parameters.path)
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

            if len(object_parameters.rotation) != 0 :
                _object.rotation_euler.x = object_parameters.rotation[0]
                _object.rotation_euler.y = object_parameters.rotation[1]
                _object.rotation_euler.z = object_parameters.rotation[2]

            bpy.context.view_layer.update()
            bpy.context.scene.collection.objects.unlink(_object)
            bpy.data.collections["Object_Medium"].objects.link(_object)

        if object_parameters.type == "sphere" :
            bpy.ops.mesh.primitive_ico_sphere_add(radius=object_parameters.radius, location=object_parameters.location,
                                                  subdivisions=object_parameters.subdivisions)
            _object = bpy.context.active_object
            _object.name = "Object"
            bpy.context.view_layer.update()
            bpy.ops.object.shade_smooth()
            bpy.context.view_layer.update()

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
            bpy.ops.mesh.primitive_cube_add(size=medium_parameters.scale, location=medium_parameters.location)
            refractive_medium = bpy.context.active_object
            refractive_medium.name = "Refractive Medium"

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
            raise ValueError('Non-perspective cameras not supported')
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
                    for light in light_cam :
                       cam_light_directions.append(light.direction)
                    data.update({"camera_{}".format(k):cam_light_directions})

            f = open(self.scene_parameters.output_path+"/lights.pkl", 'wb')
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()


        print('\033[93m' + "SAVE LIGHTS DONE\n\n" + '\033[0m')



    def save_cam_params(self):

        _all_cameras = self._scene_data["all_cams"]
        K = self.get_calibration_matrix_K_from_blender(_all_cameras[0])
        _all_R = np.zeros((3, 3, len(_all_cameras)))
        _all_T = np.zeros((3, len(_all_cameras)))
        for k, cam_k in enumerate(_all_cameras):
            R, T = self.get_3x4_RT_matrix_from_blender(cam_k)
            _all_R[:, :, k] = R
            _all_T[:, k] = T

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

    def compositing_real(self,add_mask=True):
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
        output_node.location = 400, 400
        output_node.format.file_format = "JPEG"

        # link nodes
        links = tree.links
        link = links.new(image_node.outputs["Image"], comp_node.inputs[0])
        if add_mask :
            link = links.new(image_node.outputs["IndexOB"], output_node.inputs[0])

        return output_node

    def setup_real(self):
        bpy.context.scene.cycles.samples = 256
        _object = self._scene_data["object"]
        _refractive_medium = self._scene_data["refractive_medium"]
        _all_materials = self._scene_data["all_materials"]
        _object.active_material = _all_materials["object"]
        _refractive_medium.active_material = _all_materials["refractive_medium"]
        _all_cameras = self._scene_data["all_cams"]

    def apply_rendering_params(self):

        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.scene.render.image_settings.color_depth = '16'
        bpy.context.scene.render.image_settings.compression = 15
        bpy.context.scene.view_settings.view_transform = 'Standard'
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.use_preview_denoising = True
        bpy.context.scene.cycles.use_denoising = False
        bpy.context.scene.cycles.denoiser = 'OPTIX'
        bpy.context.scene.cycles.preview_samples = 12
        bpy.context.scene.view_layers["ViewLayer"].use_pass_transmission_direct = True
        bpy.context.scene.view_layers["ViewLayer"].use_pass_object_index = True
        bpy.context.scene.view_layers["ViewLayer"].use_pass_object_index = True
        bpy.context.scene.view_layers["ViewLayer"].use_pass_cryptomatte_object = True
        bpy.context.scene.view_layers["ViewLayer"].use_pass_cryptomatte_asset = True
        bpy.context.scene.view_layers["ViewLayer"].use_pass_cryptomatte_material = True
        if self.scene_parameters.stereo_photometry :
            bpy.context.scene.cycles.max_bounces = 200
            bpy.context.scene.cycles.glossy_bounces = 0
            bpy.context.scene.cycles.transmission_bounces = 100
            bpy.context.scene.cycles.diffuse_bounces = 100
            bpy.context.scene.cycles.transparent_max_bounces = 0
            bpy.context.scene.display_settings.display_device = 'None'
        else :
            bpy.context.scene.cycles.max_bounces = 200
            bpy.context.scene.cycles.glossy_bounces = 100
            bpy.context.scene.cycles.transmission_bounces = 40
            bpy.context.scene.cycles.diffuse_bounces = 40
            bpy.context.scene.cycles.transparent_max_bounces = 10
            bpy.context.scene.cycles.volume_bounces = 10


    def render_object_with_amber(self):

        bpy.context.scene.cycles.samples = 256
        _object = self._scene_data["object"]
        _refractive_medium = self._scene_data["refractive_medium"]
        _all_materials = self._scene_data["all_materials"]
        _object.active_material = _all_materials["object"]
        _refractive_medium.active_material = _all_materials["refractive_medium"]
        _all_cameras = self._scene_data["all_cams"]
        _all_lights = self._scene_data["all_lights"]

        if self.scene_parameters.lights.fixed_light :
            output_node = self.compositing_real()
            output_node.base_path = self.scene_parameters.output_path+"medium_masks/"
            _refractive_medium.pass_index = 1

            for cam_num,cam_k in enumerate(_all_cameras):
                print('\033[93m' + "OBJECT WITH AMBER [{}/{}]\n".format(cam_num+1, len(_all_cameras)) + '\033[0m')
                bpy.context.scene.camera = cam_k
                output_node.file_slots[0].path = f'{cam_k.name}_0'
                output_node.format.file_format = "PNG"
                bpy.context.scene.render.filepath = self.scene_parameters.output_path+"images_with/"+f'{cam_k.name}_0.png'
                bpy.ops.render.render(write_still=1)
            _refractive_medium.pass_index = 0

        else :
            _refractive_medium.pass_index = 1
            for cam_num, cam_k in enumerate(_all_cameras):
                output_node = self.compositing_real()
                output_node.base_path = self.scene_parameters.output_path + "medium_masks/"
                print('\033[93m' + "OBJECT WITH AMBER [{}/{}]\n".format(cam_num + 1, len(_all_cameras)) + '\033[0m')
                bpy.context.scene.camera = cam_k
                for light_num in range(len(_all_lights[cam_num])):
                    light = _all_lights[cam_num][light_num]
                    light.hide_render = False
                    output_node.file_slots[0].path = f'{cam_k.name}_'
                    output_node.format.file_format = "PNG"
                    bpy.context.scene.render.filepath = self.scene_parameters.output_path + "images_with/" + f'{cam_k.name}_{light_num}.png'
                    bpy.ops.render.render(write_still=1)
                    if light_num == 0 :
                        output_node = self.compositing_real(False)
                    light.hide_render = True
            _refractive_medium.pass_index = 0

    def render_object_without_amber(self):

        bpy.context.scene.cycles.samples = 256
        _object = self._scene_data["object"]
        _refractive_medium = self._scene_data["refractive_medium"]
        _refractive_medium.hide_render = True
        _all_materials = self._scene_data["all_materials"]
        _object.active_material = _all_materials["object"]
        _refractive_medium.active_material = _all_materials["refractive_medium"]
        _all_cameras = self._scene_data["all_cams"]
        _all_lights = self._scene_data["all_lights"]

        if self.scene_parameters.lights.fixed_light :
            _object.pass_index = 1
            output_node = self.compositing_real()
            output_node.base_path = self.scene_parameters.output_path + "images_masks_without/"
            for cam_num,cam_k in enumerate(_all_cameras):
                print('\033[93m' + "OBJECT WITHOUT AMBER [{}/{}]\n".format(cam_num+1,len(_all_cameras)) + '\033[0m')
                bpy.context.scene.camera = cam_k
                output_node.file_slots[0].path = f'{cam_k.name}_0'
                output_node.format.file_format = "PNG"
                bpy.context.scene.render.filepath = self.scene_parameters.output_path+"images_without/" + f'{cam_k.name}_0.png'
                bpy.ops.render.render(write_still=1)
            _refractive_medium.hide_render = False
            _object.pass_index = 0

        else :
            _object.pass_index = 1
            for cam_num, cam_k in enumerate(_all_cameras):
                output_node = self.compositing_real()
                output_node.base_path = self.scene_parameters.output_path + "images_masks_without/"
                print('\033[93m' + "OBJECT WITHOUT AMBER [{}/{}]\n".format(cam_num + 1, len(_all_cameras)) + '\033[0m')
                bpy.context.scene.camera = cam_k
                for light_num in range(len(_all_lights[cam_num])):
                    light = _all_lights[cam_num][light_num]
                    light.hide_render = False
                    output_node.file_slots[0].path = f'{cam_k.name}_'
                    output_node.format.file_format = "PNG"
                    bpy.context.scene.render.filepath = self.scene_parameters.output_path + "images_without/" + f'{cam_k.name}_{light_num}.png'
                    bpy.ops.render.render(write_still=1)
                    if light_num == 0 :
                        output_node = self.compositing_real(False)
                    light.hide_render = True
            _refractive_medium.hide_render = False
            _object.pass_index = 0

    def create_output_folders(self):
        if not os.path.exists(self.scene_parameters.output_path):
            os.mkdir(self.scene_parameters.output_path)
        if not os.path.exists(self.scene_parameters.output_path+"images_without/"):
            os.mkdir(self.scene_parameters.output_path+"images_without/")
        if not os.path.exists(self.scene_parameters.output_path+"images_masks_without/"):
            os.mkdir(self.scene_parameters.output_path+"images_masks_without/")
        if not os.path.exists(self.scene_parameters.output_path+"images_with/"):
            os.mkdir(self.scene_parameters.output_path+"images_with/")
        if not os.path.exists(self.scene_parameters.output_path+"images_masks_with/"):
            os.mkdir(self.scene_parameters.output_path+"images_masks_with/")
        if not os.path.exists(self.scene_parameters.output_path+"medium_masks/"):
            os.mkdir(self.scene_parameters.output_path+"medium_masks/")

    def modal(self, context, event):

        if event.type in {'F1'}:
            if self.first_F1 :
                self.first_F1 = False
                self.create_output_folders()
                self._scene_data = self.generate_fixed_scene()
                self.nb_cam = len(self._scene_data["all_cams"])
                if self.scene_parameters.render_with_obj_mask :
                    self.l_object.append([self._scene_data["object"]])
                easy_keys.run(push_f5())

        if event.type in {'F5'}:
            if (not self.first_F1) and self.first_F5 :
                self.first_F5 = False

                if self.scene_parameters.render_without :
                    self.render_object_without_amber()
                if  self.scene_parameters.render_with :
                    self.render_object_with_amber()

                self.save_cam_params()
                self.save_interface_data()
                self.save_lights_params()

                self.setup_real()
                if self.scene_parameters.save_scene :
                    bpy.ops.wm.save_as_mainfile(filepath=self.scene_parameters.output_path + "/scene.blend")
                self.activate_film(True)
                self.cryptomatte_node,self.viewer_node = self.compositing_cryptomatte()
                easy_keys.run(push_f7())


        if event.type in {'F7'}:
            if (not self.first_F1) and self.not_f7:
                if self.ind_render < len(self.l_render) :
                    if self.ind_cam_k < self.nb_cam  :
                        print('\033[93m' + "OBJECT AMBER MASK [{}/{}]\n".format(self.ind_cam_k + 1,self.nb_cam) + '\033[0m')
                        self.cryptomatte_node.matte_id = self.l_object[self.ind_render][0].name
                        cam_k = self._scene_data["all_cams"][self.ind_cam_k]
                        bpy.context.scene.camera = cam_k
                        bpy.ops.render.render(write_still=0)
                        self.ind_cam_k += 1
                        self.ind_object = 0
                        self.not_f7 = False
                        self.not_f8 = True
                        easy_keys.run(push_f8())
                    else :
                        bpy.app.use_event_simulate = False
                        bpy.context.window.workspace = bpy.data.workspaces['Layout']
                        self.activate_film(False)
                        self.global_white_light(False)

                else :
                    bpy.app.use_event_simulate = False
                    bpy.context.window.workspace = bpy.data.workspaces['Layout']
                    self.activate_film(False)

        if event.type in {'F8'}:
            if (not self.first_F1) and self.not_f8 :
                bpy.data.images["Viewer Node"].save_render(filepath=self.output_render[self.ind_render]+f"camera_{self.ind_cam_k-1}_.png")
                easy_keys.run(push_f7())
                self.not_f7 = True
                self.not_f8 = False

        return {'PASS_THROUGH'}

    def execute(self, context):
        wm = context.window_manager
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)

def register():
    bpy.utils.register_class(ModalTimerOperator)

def unregister():
    bpy.utils.unregister_class(ModalTimerOperator)

def push_f1():
    window = bpy.context.window
    e = easy_keys.EventGenerate(window)
    yield e.f1()
    yield easy_keys.Finish
def push_f5():
    window = bpy.context.window
    e = easy_keys.EventGenerate(window)
    yield e.f5()
    yield easy_keys.Finish
def push_f7():
    window = bpy.context.window
    e = easy_keys.EventGenerate(window)
    yield e.f7()
    yield easy_keys.Finish
def push_f8():
    window = bpy.context.window
    e = easy_keys.EventGenerate(window)
    yield e.f8()
    yield easy_keys.Finish


if __name__ == "__main__":
    register()
    scene = generate_blender_scene()
    bpy.ops.wm.modal_timer_operator()
    bpy.context.window.workspace = bpy.data.workspaces['Compositing']
    random.seed(1)
    easy_keys.tweak_preferences(bpy.context.preferences)
    easy_keys.run(push_f1())
