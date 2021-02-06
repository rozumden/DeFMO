""" render_fmo.py renders obj file to rgb image with fmo model

Aviable function:
- clear_mash: delete all the mesh in the secene
- scene_setting_init: set scene configurations
- node_setting_init: set node configurations
- render: render rgb image for one obj file and one viewpoint
- render_obj: wrapper function for render() render 
- init_all: a wrapper function, initialize all configurations                          
= set_image_path: reset defualt image output folder

author baiyu
modified by rozumden
"""
import sys
import os
import random
import pickle
import bpy
import glob
import numpy as np
from mathutils import Vector
from mathutils import Euler
import cv2
from PIL import Image
from skimage.draw import line_aa
from scipy import signal
from skimage.measure import regionprops
# import moviepy.editor as mpy
from array2gif import write_gif

abs_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(abs_path))

from render_helper import *
from settings import *
import settings
import pdb

def renderTraj(pars, H):
    ## Input: pars is either 2x2 (line) or 2x3 (parabola)
    if pars.shape[1] == 2:
        pars = np.concatenate( (pars, np.zeros((2,1))),1)
        ns = 2
    else:
        ns = 5
    ns = np.max([2, ns])
    rangeint = np.linspace(0,1,ns)
    for timeinst in range(rangeint.shape[0]-1):
        ti0 = rangeint[timeinst]
        ti1 = rangeint[timeinst+1]
        start = pars[:,0] + pars[:,1]*ti0 + pars[:,2]*(ti0*ti0)
        end = pars[:,0] + pars[:,1]*ti1 + pars[:,2]*(ti1*ti1)
        start = np.round(start).astype(np.int32)
        end = np.round(end).astype(np.int32)
        rr, cc, val = line_aa(start[0], start[1], end[0], end[1])
        valid = np.logical_and(np.logical_and(rr < H.shape[0], cc < H.shape[1]), np.logical_and(rr > 0, cc > 0))
        rr = rr[valid]
        cc = cc[valid]
        val = val[valid]
        if len(H.shape) > 2:
            H[rr, cc, 0] = 0
            H[rr, cc, 1] = 0
            H[rr, cc, 2] = val
        else:
            H[rr, cc] = val 
    return H

def open_log(temp_folder = g_temp): # redirect output to log file
    logfile = os.path.join(temp_folder,'blender_render.log')
    try:
        os.remove(logfile)
    except OSError:
        pass
    open(logfile, 'a').close()
    old = os.dup(1)
    sys.stdout.flush()
    os.close(1)
    os.open(logfile, os.O_WRONLY)
    return old

def close_log(old): # disable output redirection
    os.close(1)
    os.dup(old)
    os.close(old)

def clear_mesh():
    """ clear all meshes in the secene

    """
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            obj.select = True
    bpy.ops.object.delete()
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)

    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)

    for block in bpy.data.textures:
        if block.users == 0:
            bpy.data.textures.remove(block)

    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)

def scene_setting_init(use_gpu):
    """initialize blender setting configurations

    """
    sce = bpy.context.scene.name
    bpy.data.scenes[sce].render.engine = g_engine_type
    bpy.data.scenes[sce].cycles.film_transparent = g_use_film_transparent

    #output
    bpy.data.scenes[sce].render.image_settings.color_mode = g_rgb_color_mode
    bpy.data.scenes[sce].render.image_settings.color_depth = g_rgb_color_depth
    bpy.data.scenes[sce].render.image_settings.file_format = g_rgb_file_format
    bpy.data.scenes[sce].render.use_overwrite = g_depth_use_overwrite
    bpy.data.scenes[sce].render.use_file_extension = g_depth_use_file_extension 

    if g_ambient_light:
        world = bpy.data.worlds['World']
        world.use_nodes = True
        bg = world.node_tree.nodes['Background']
        bg.inputs[0].default_value[:3] = g_bg_color 
        bg.inputs[1].default_value = 1.0
        
    #dimensions
    bpy.data.scenes[sce].render.resolution_x = g_resolution_x
    bpy.data.scenes[sce].render.resolution_y = g_resolution_y
    bpy.data.scenes[sce].render.resolution_percentage = g_resolution_percentage

    if use_gpu:
        bpy.data.scenes[sce].render.engine = 'CYCLES' #only cycles engine can use gpu
        bpy.data.scenes[sce].render.tile_x = g_hilbert_spiral
        bpy.data.scenes[sce].render.tile_x = g_hilbert_spiral
        bpy.context.user_preferences.addons['cycles'].preferences.devices[0].use = False
        bpy.context.user_preferences.addons['cycles'].preferences.devices[1].use = True
        ndev = len(bpy.context.user_preferences.addons['cycles'].preferences.devices)
        print('Number of devices {}'.format(ndev))
        for ki in range(2,ndev):
            bpy.context.user_preferences.addons['cycles'].preferences.devices[ki].use = False
        bpy.context.user_preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
        # bpy.types.CyclesRenderSettings.device = 'GPU'
        bpy.data.scenes[sce].cycles.device = 'GPU'

def node_setting_init():
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links
    for node in tree.nodes:
        tree.nodes.remove(node)
    render_layer_node = tree.nodes.new('CompositorNodeRLayers')
    image_output_node = tree.nodes.new('CompositorNodeOutputFile')
    image_output_node.base_path = g_syn_rgb_folder

    links.new(render_layer_node.outputs[0], image_output_node.inputs[0])

    # image_output_node = bpy.context.scene.node_tree.nodes[1]
    image_output_node.base_path = g_temp
    image_output_node.file_slots[0].path = 'image-######.png' # blender placeholder #
     

def render(obj_path, viewpoint, temp_folder):
    """render rbg image 

    render a object rgb image by a given camera viewpoint and
    choose random image as background, only render one image
    at a time.

    Args:
        obj_path: a string variable indicate the obj file path
        viewpoint: a vp parameter(contains azimuth,elevation,tilt angles and distance)
    """
    vp = viewpoint
    cam_location = camera_location(vp.azimuth, vp.elevation, vp.distance)
    cam_rot = camera_rot_XYZEuler(vp.azimuth, vp.elevation, vp.tilt)
    cam_obj = bpy.data.objects['Camera']
    cam_obj.location[0] = cam_location[0]
    cam_obj.location[1] = cam_location[1]
    cam_obj.location[2] = cam_location[2]
    cam_obj.rotation_euler[0] = cam_rot[0]
    cam_obj.rotation_euler[1] = cam_rot[1]
    cam_obj.rotation_euler[2] = cam_rot[2]
    if not os.path.exists(g_syn_rgb_folder):
        os.mkdir(g_syn_rgb_folder)

    obj = bpy.data.objects['model_normalized']
   
    ni = g_fmo_steps
    maxlen = 0.5
    maxrot = 1.57/6
    tri = 0
    # rot_base = np.array([math.pi/2,0,0])
    while tri <= g_max_trials:
        do_repeat = False
        tri += 1
        if not g_apply_texture:
            for oi in range(len(bpy.data.objects)): 
                if bpy.data.objects[oi].type == 'CAMERA' or bpy.data.objects[oi].type == 'LAMP':
                    continue
                for tempi in range(len(bpy.data.objects[oi].data.materials)): 
                    if bpy.data.objects[oi].data.materials[tempi].alpha != 1.0:
                        return True, True ## transparent object

        los_start = Vector((random.uniform(-maxlen/10, maxlen/10), random.uniform(-maxlen, maxlen), random.uniform(-maxlen, maxlen)))
        loc_step = Vector((random.uniform(-maxlen/10, maxlen/10), random.uniform(-maxlen, maxlen), random.uniform(-maxlen, maxlen)))/ni
        
        rot_base = np.array((random.uniform(0, 2*math.pi), random.uniform(0, 2*math.pi), random.uniform(0, 2*math.pi)))
        rot_step = np.array((random.uniform(-maxrot, maxrot), random.uniform(-maxrot, maxrot), random.uniform(-maxrot, maxrot)))/ni
        old = open_log(temp_folder)       
        for ki in [0, ni-1]+list(range(1,ni-1)):
            for oi in range(len(bpy.data.objects)): 
                if bpy.data.objects[oi].type == 'CAMERA' or bpy.data.objects[oi].type == 'LAMP':
                    continue
                bpy.data.objects[oi].location = los_start + loc_step*ki
                bpy.data.objects[oi].rotation_euler = Euler(rot_base + (rot_step*ki))
            bpy.context.scene.frame_set(ki + 1)
            bpy.ops.render.render(write_still=True) #start rendering
            if ki == 0 or ki == (ni-1):
                Mt = cv2.imread(os.path.join(bpy.context.scene.node_tree.nodes[1].base_path,'image-{:06d}.png'.format(ki+1)),cv2.IMREAD_UNCHANGED)[:,:,-1] > 0
                is_border = ((Mt[0,:].sum()+Mt[-1,:].sum()+Mt[:,0].sum()+Mt[:,-1].sum()) > 0) or Mt.sum()==0
                if is_border:
                    if ki == 0:
                        close_log(old)
                        return False, True ## sample different starting viewpoint
                    else:
                        do_repeat = True ## just sample another motion direction
                if do_repeat:
                    break
        close_log(old)
        if do_repeat == False:
            break
    if do_repeat: ## sample different starting viewpoint
        return False, True
    return False, False

def make_fmo(path, gt_path, video_path):
    n_im = 5
    background_images = os.listdir(g_background_image_path)
    seq_name = random.choice(background_images)
    seq_images = glob.glob(os.path.join(g_background_image_path,seq_name,"*.jpg"))
    if len(seq_images) <= n_im:
        seq_images = glob.glob(os.path.join(g_background_image_path,seq_name,"*.png"))
    seq_images.sort()
    bgri = random.randint(n_im,len(seq_images)-1)
    bgr_path = seq_images[bgri]

    B0 = cv2.imread(bgr_path)/255
    B = cv2.resize(B0, dsize=(int(g_resolution_x*g_resolution_percentage/100), int(g_resolution_y*g_resolution_percentage/100)), interpolation=cv2.INTER_CUBIC)
    B[B > 1] = 1
    B[B < 0] = 0
    FH = np.zeros(B.shape)
    MH = np.zeros(B.shape[:2])
    pars = np.array([[(B.shape[0]-1)/2-1, (B.shape[1]-1)/2-1], [1.0, 1.0]]).T
    FM = np.zeros(B.shape[:2]+(4,g_fmo_steps,))
    centroids = np.zeros((2,g_fmo_steps))
    for ki in range(g_fmo_steps):
        FM[:,:,:,ki] = cv2.imread(os.path.join(gt_path,'image-{:06d}.png'.format(ki+1)),cv2.IMREAD_UNCHANGED)/g_rgb_color_max
        props = regionprops((FM[:,:,-1,ki]>0).astype(int))
        if len(props) != 1:
            return False
        centroids[:,ki] = props[0].centroid
    for ki in range(g_fmo_steps):
        F = FM[:,:,:-1,ki]*FM[:,:,-1:,ki]
        M = FM[:,:,-1,ki]
        if ki < g_fmo_steps-1:
            pars[:,1] = centroids[:,ki+1] - centroids[:,ki]
        H = renderTraj(pars, np.zeros(B.shape[:2]))
        H /= H.sum()*g_fmo_steps
        for kk in range(3): 
            FH[:,:,kk] += signal.fftconvolve(H, F[:,:,kk], mode='same')
        MH += signal.fftconvolve(H, M, mode='same')
    Im = FH + (1 - MH)[:,:,np.newaxis]*B
    Im[Im > 1] = 1
    Im[Im < 0] = 0
    if g_skip_low_contrast:
        Diff = np.sum(np.abs(Im - B),2)
        meanval = np.mean(Diff[MH > 0.05])
        print("Contrast {}".format(meanval))
        if meanval < 0.2:
            return False
    if g_skip_small:
        sizeper = np.sum(MH > 0.01)/(MH.shape[0]*MH.shape[1])
        print("Size percentage {}".format(sizeper))
        if sizeper < 0.05:
            return False
   
    Im = Im[:,:,[2,1,0]]
    Ims = Image.fromarray((Im * 255).astype(np.uint8))

    Ims.save(path)

    Ball = np.zeros(B.shape+(n_im,))
    Ball[:,:,:,0] = B
    for ki in range(1,n_im):
        bgrki_path = seq_images[bgri-ki]
        Ball[:,:,:,ki] = cv2.resize(cv2.imread(bgrki_path)/255, dsize=(int(g_resolution_x*g_resolution_percentage/100), int(g_resolution_y*g_resolution_percentage/100)), interpolation=cv2.INTER_CUBIC)
    Ball[Ball > 1] = 1
    Ball[Ball < 0] = 0
    Bmed = np.median(Ball,3)
    Image.fromarray((B[:,:,[2,1,0]] * 255).astype(np.uint8)).save(os.path.join(gt_path,'bgr.png'))
    Image.fromarray((Bmed[:,:,[2,1,0]] * 255).astype(np.uint8)).save(os.path.join(gt_path,'bgr_med.png'))

    # Ims.save(os.path.join(g_temp,"I.png"))
    # Image.fromarray((FH * 255)[:,:,[2,1,0]].astype(np.uint8)).save(os.path.join(g_temp,"FH.png"))
    # Image.fromarray((MH * 255).astype(np.uint8)).save(os.path.join(g_temp,"MH.png"))
    # Image.fromarray((M * 255).astype(np.uint8)).save(os.path.join(g_temp,"M.png"))
    # Image.fromarray((F * 255)[:,:,[2,1,0]].astype(np.uint8)).save(os.path.join(g_temp,"F.png"))
    # Image.fromarray((B0 * 255)[:,:,[2,1,0]].astype(np.uint8)).save(os.path.join(g_temp,"B.png"))

    if False:
        Fwr = FM[:,:,:-1,:] * FM[:,:,-1:,:] + 1 * (1 - FM[:,:,-1:,:])
        Fwr = (Fwr * 255).astype(np.uint8)
        # Fwr[np.repeat(FM[:,:,-1:,:]==0,3,2)]=255 
        out = cv2.VideoWriter(video_path,cv2.VideoWriter_fourcc(*"MJPG"), 6, (F.shape[1],F.shape[0]),True)
        for ki in range(g_fmo_steps):
            out.write(Fwr[:,:,:,ki])
        out.release()
        
    return True

def render_obj(obj_path, path, objid, obj_name, temp_folder):
    """ render one obj file by a given viewpoint list
    a wrapper function for render()

    Args:
        obj_path: a string variable indicate the obj file path
    """
    vps_path = random.sample(g_view_point_file, 1)[0]
    vps = list(load_viewpoint(vps_path))
    random.shuffle(vps)
    save_path = os.path.join(path,"{}_{:04d}.png".format(obj_name,objid))
    gt_path = os.path.join(path,"GT","{}_{:04d}".format(obj_name,objid))
    video_path = os.path.join(path,"{}_{:04d}.avi".format(obj_name,objid))
    if not os.path.exists(gt_path):
        os.mkdir(gt_path)
    image_output_node = bpy.context.scene.node_tree.nodes[1]
    image_output_node.base_path = gt_path

    for imt in bpy.data.images:
        bpy.data.images.remove(imt)

    if g_apply_texture:
        for oi in range(len(bpy.data.objects)): 
            if bpy.data.objects[oi].type == 'CAMERA' or bpy.data.objects[oi].type == 'LAMP':
                continue
            bpy.context.scene.objects.active = bpy.data.objects[oi]
            # pdb.set_trace()
            # for m in bpy.data.materials:
            #     bpy.data.materials.remove(m)
            # bpy.ops.object.material_slot_remove()
            
            bpy.ops.object.editmode_toggle()
            bpy.ops.uv.cube_project()
            bpy.ops.object.editmode_toggle()

            texture_images = os.listdir(g_texture_path)
            texture = random.choice(texture_images)
            tex_path = os.path.join(g_texture_path,texture)
               
            # mat = bpy.data.materials.new(texture)
            # mat.use_nodes = True
            # nt = mat.node_tree
            # nodes = nt.nodes
            # links = nt.links

            # # Image Texture
            # textureNode = nodes.new("ShaderNodeTexImage")
            # textureNode.image = bpy.data.images.load(tex_path)
            # links.new(nodes['Diffuse BSDF'].inputs['Color'],   textureNode.outputs['Color'])

            # mat.specular_intensity = 0

            # bpy.data.objects[oi].active_material = mat
            # print(bpy.data.objects[oi].active_material)
            for mat in bpy.data.materials:
                nodes = mat.node_tree.nodes
                links = mat.node_tree.links
                textureNode = nodes.new("ShaderNodeTexImage")
                textureNode.image = bpy.data.images.load(tex_path)
                links.new(nodes['Diffuse BSDF'].inputs['Color'],   textureNode.outputs['Color'])
            # print(bpy.data.objects[oi].active_material)
            
    tri = 0
    while tri <= g_max_trials:
        tri += 1
        vp = random.sample(vps, 1)[0]
        
        sample_different_object, sample_different_vp = render(obj_path, vp, temp_folder)

        if sample_different_vp:
            if sample_different_object:
                print('Transparent object!')
                return False
            print('Rendering failed, repeating')
            continue
        success = make_fmo(save_path, gt_path, video_path)
        if success:
            return True
        print('Making FMO failed, repeating')
    return False

def init_all():
    """init everything we need for rendering
    an image
    """
    scene_setting_init(g_gpu_render_enable)
    node_setting_init()
    cam_obj = bpy.data.objects['Camera']
    cam_obj.rotation_mode = g_rotation_mode

    if g_render_light:
        bpy.data.objects['Lamp'].data.energy = 50
        bpy.ops.object.lamp_add(type='SUN')    
        bpy.data.objects['Sun'].data.energy = 5


### YOU CAN WRITE YOUR OWN IMPLEMENTATION TO GENERATE DATA

init_all()

argv = sys.argv
argv = argv[argv.index("--") + 1:]  
start_index = int(argv[0])
step_index = int(argv[1])
print('Start index {}, step index {}'.format(start_index, step_index))
temp_folder = g_syn_rgb_folder+g_render_objs[start_index]+'/'

for obj_name in g_render_objs[start_index:(start_index+step_index)]:
    print("Processing object {}".format(obj_name))
    obj_folder = os.path.join(g_syn_rgb_folder, obj_name)
    if not os.path.exists(obj_folder):
        os.makedirs(obj_folder)
    if not os.path.exists(os.path.join(obj_folder,"GT")):
        os.mkdir(os.path.join(obj_folder,"GT"))

    num = g_shapenet_categlory_pair[obj_name]
    search_path = os.path.join(g_shapenet_path, num, '**','*.obj')
    pathes = glob.glob(search_path, recursive=True)
    random.shuffle(pathes)
    objid = 1
    tri = 0
    while objid <= g_number_per_category:
        print("           instance {}".format(objid))
        clear_mesh()
        path = random.sample(pathes, 1)[0]
        old = open_log(temp_folder)
        bpy.ops.import_scene.obj(filepath=path, axis_forward='-Z', axis_up='Y', filter_glob="*.obj;*.mtl", use_split_groups=False, use_split_objects=True)
        # bpy.ops.import_scene.obj(filepath=path)
        close_log(old)
        #combine_objects()
        #scale_objects(0.5)
        result = render_obj(path, obj_folder, objid, obj_name, temp_folder)
        if result:
            objid += 1
            tri = 0
        else:
            print('Error! Rendering another object from the category!')
            tri += 1
            if tri > g_max_trials:
                print('No object find in the category!!!!!!!!!')
                break