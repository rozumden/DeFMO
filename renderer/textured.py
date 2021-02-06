import bpy
import os
import numpy as np
import json
from os.path import join
import argparse
import sys
import pickle
import time
from mathutils import Matrix

## taken from https://github.com/krematas/blender_render
## author: Konstantinos Rematas

def remove_materials_except(saved_materials=['Carpet', 'Floor']):
  count = 0
  for m in bpy.data.materials:
    if m.name not in saved_materials:
      bpy.data.materials.remove(m)
      count += 1
    else:
      print('Material {0} saved'.format(m.name))
  return count


#---------------------------------------------------------------
# 3x4 P matrix from Blender camera
#---------------------------------------------------------------

# Build intrinsic camera parameters from Blender camera data
#
# See notes on this in
# blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model
def get_calibration_matrix_K_from_blender(camd):
  f_in_mm = camd.lens
  scene = bpy.context.scene
  resolution_x_in_px = scene.render.resolution_x
  resolution_y_in_px = scene.render.resolution_y
  scale = scene.render.resolution_percentage / 100
  sensor_width_in_mm = camd.sensor_width
  sensor_height_in_mm = camd.sensor_height
  pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
  if (camd.sensor_fit == 'VERTICAL'):
    # the sensor height is fixed (sensor fit is horizontal),
    # the sensor width is effectively changed with the pixel aspect ratio
    s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
    s_v = resolution_y_in_px * scale / sensor_height_in_mm
  else: # 'HORIZONTAL' and 'AUTO'
    # the sensor width is fixed (sensor fit is horizontal),
    # the sensor height is effectively changed with the pixel aspect ratio
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    s_u = resolution_x_in_px * scale / sensor_width_in_mm
    s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm
  # Parameters of intrinsic calibration matrix K
  alpha_u = f_in_mm * s_u
  alpha_v = f_in_mm * s_v
  u_0 = resolution_x_in_px * scale / 2
  v_0 = resolution_y_in_px * scale / 2
  skew = 0 # only use rectangular pixels
  K = Matrix(
      ((alpha_u, skew,    u_0),
       (    0  , alpha_v, v_0),
       (    0  , 0,        1 )))
  return K

# Returns camera rotation and translation matrices from Blender.
#
# There are 3 coordinate systems involved:
#    1. The World coordinates: "world"
#       - right-handed
#    2. The Blender camera coordinates: "bcam"
#       - x is horizontal
#       - y is up
#       - right-handed: negative z look-at direction
#    3. The desired computer vision camera coordinates: "cv"
#       - x is horizontal
#       - y is down (to align to the actual pixel coordinates
#         used in digital images)
#       - right-handed: positive z look-at direction
def get_3x4_RT_matrix_from_blender(cam):
  # bcam stands for blender camera
  R_bcam2cv = Matrix(
      ((1, 0,  0),
       (0, -1, 0),
       (0, 0, -1)))
  # Transpose since the rotation is object rotation,
  # and we want coordinate rotation
  # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
  # T_world2bcam = -1*R_world2bcam * location
  #
  # Use matrix_world instead to account for all constraints
  location, rotation = cam.matrix_world.decompose()[0:2]
  R_world2bcam = rotation.to_matrix().transposed()
  # Convert camera location to translation vector used in coordinate changes
  # T_world2bcam = -1*R_world2bcam*cam.location
  # Use location from matrix_world to account for constraints:
  T_world2bcam = -1*R_world2bcam @ location
  # Build the coordinate transform matrix from world to computer vision camera
  R_world2cv = R_bcam2cv@R_world2bcam
  T_world2cv = R_bcam2cv@T_world2bcam
  # put into 3x4 matrix
  RT = Matrix((
      R_world2cv[0][:] + (T_world2cv[0],),
      R_world2cv[1][:] + (T_world2cv[1],),
      R_world2cv[2][:] + (T_world2cv[2],)
  ))
  return RT



argv = sys.argv
argv = argv[argv.index("--") + 1:]

parser = argparse.ArgumentParser()
parser.add_argument('--start', dest='start_frame', type=int, default=0)
parser.add_argument('--end', dest='end_frame', type=int, default=10)
parser.add_argument('--output_dir', default='', help='Dataset to train on')
parser.add_argument('--output_info_file', default='', help='Dataset to train')
parser.add_argument('--input_info_file', default='', help='Dataset to train')
parser.add_argument('--n_samples', dest='n_samples', type=int, default=25)
parser.add_argument('--mode', default='train', help='Dataset to train on')
parser.add_argument('--scale', type=float, default=1.0)
parser.add_argument('--transl_area', type=float, default=0.4)

args = parser.parse_known_args(argv)[0]

#===============================================================================
print(args)
model_s = args.start_frame
model_e = args.end_frame
render_path = args.output_dir
n_samples = args.n_samples

scale = args.scale
transl_area = args.transl_area

#===============================================================================
context = bpy.context
scene = bpy.context.scene
tree = bpy.context.scene.node_tree

cam = bpy.data.objects['Camera']
lamp = bpy.data.objects['Lamp']

#===============================================================================
if not os.path.exists(render_path):
  os.mkdir(render_path)

#===============================================================================
path_to_textures = '/PATH_TO_TEXTURES/textures_train'
textures_train = [f for f in os.listdir(path_to_textures)]
for f in textures_train:
  _ = bpy.ops.image.open(filepath=os.path.join(path_to_textures, f))

path_to_textures = '/PATH_TO_TEXTURES/textures_test'
textures_test = [f for f in os.listdir(path_to_textures)]
for f in textures_test:
  _ = bpy.ops.image.open(filepath=os.path.join(path_to_textures, f))

#===============================================================================
experiment_data = pickle.load(open(args.input_info_file, 'rb'))

n_renders = len(experiment_data)
#===============================================================================
image_counter = 0
data2save = []
for i_exp in range(model_s, model_e):
  cur_obj = experiment_data[i_exp]['obj']
  cur_ex = experiment_data[i_exp]['example_id']

  print('=====================================================================')
  print('Model: {0} ({1}/{2})\n\n'.format(cur_ex, i_exp, len(experiment_data)))

  obj_path = os.path.join(experiment_data[i_exp]['full_obj_path'])
  bpy.ops.import_scene.obj(filepath=obj_path, axis_forward='-Z', axis_up='Y', filter_glob="*.obj;*.mtl", use_split_groups=False, use_split_objects=True)

  obj = bpy.data.objects['model_normalized']
  bpy.context.view_layer.objects.active = obj

  n_removed_materials = remove_materials_except()

  #=============================================================================
  vertex = [[vert.co.x, vert.co.y, vert.co.z] for vert in obj.data.vertices]
  vertex = np.array(vertex)
  min_pos = np.min(vertex, axis=0)
  max_pos = np.max(vertex, axis=0)
  mean_pos = (min_pos + max_pos)/2.0
  min_y = np.min(vertex[:, 2], axis=0)

  #=============================================================================

  # Assign texture to 3D model =================================================
  obj.select_set(True)
  for _ in range(n_removed_materials):
    bpy.ops.object.material_slot_remove()

  bpy.ops.object.editmode_toggle()
  bpy.ops.uv.cube_project()
  bpy.ops.object.editmode_toggle()

  # Assign texture to material
  mat = bpy.data.materials.new(name='ScriptedMaterial.0')
  mat.use_nodes = True
  bsdf = mat.node_tree.nodes['Principled BSDF']

  tex = mat.node_tree.nodes.new('ShaderNodeTexImage')
  mat.node_tree.links.new(bsdf.inputs['Base Color'], tex.outputs['Color'])

  image_tex = mat.node_tree.nodes.feature_get('Image Texture')

  tex_path = experiment_data[i_exp]['texture']
  image_tex.image = bpy.data.images[os.path.basename(tex_path)]

  bsdf.inputs['Specular'].default_value = 0.0
  bsdf.inputs['Roughness'].default_value = 0.0

  # obj.data.materials.append(mat) #add the material to the object
  obj.active_material = mat
  #=============================================================================

  obj.location.z -= min_pos[1]*scale
  obj.scale = (scale, scale, scale)

  floor_color = list(np.random.rand(4))
  bpy.data.materials['FloorKR'].node_tree.nodes['Principled BSDF'].inputs[0].default_value = floor_color

  #=============================================================================
  for j in range(n_samples):
    bpy.context.scene.frame_set(np.random.randint(5, 40))
    random_angle = np.random.randint(90, 270)

    lx, ly, lz = np.random.uniform(-1.5, 1.5, 1)[0], \
                 np.random.uniform(-1.5, 1.5, 1)[0], \
                 np.random.uniform(2.5, 3.0, 1)[0]
    lamp.location = (lx, ly, lz)

    ox, oz = np.random.uniform(-transl_area, transl_area, 1)[0]*scale, \
             np.random.uniform(-transl_area, transl_area, 1)[0]*scale

    # obj = bpy.data.objects['model_normalized']
    obj.rotation_euler = (np.deg2rad(90), 0, np.deg2rad(random_angle))
    obj.location.x = ox
    obj.location.y = oz

    filename = '{0}_{1:05d}'.format(cur_ex, j)

    context.scene.render.filepath = os.path.join(render_path, filename)
    bpy.ops.render.render(write_still=True)

    K = get_calibration_matrix_K_from_blender(cam.data)
    RT = get_3x4_RT_matrix_from_blender(cam)

    info = {'filename': filename,
            'obj': cur_obj,
            'rotation': random_angle,
            'translation': [ox, 0, oz],
            'light': [lx, ly, lz],
            'elevation': np.rad2deg(cam.matrix_world.to_euler().x),
            'RT': np.array(RT),
            'K': np.array(K),
            'floor_color': floor_color}
    data2save.append(info)

    image_counter += 1

  bpy.data.meshes.remove(bpy.data.meshes['model_normalized'])

with open(join(render_path, args.output_info_file+'.p'), 'wb') as fp:
  pickle.dump(data2save, fp, protocol=pickle.HIGHEST_PROTOCOL)
