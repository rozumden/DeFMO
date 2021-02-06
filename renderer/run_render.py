"""this module execute render operation 

author baiyu
"""



import os
import sys
import subprocess
import pickle

from render_helper import *
from settings import *



if __name__ == '__main__':
    #you could also just comment these 4 lines after run it once
    #if you want to use the same models and vps instead of generate
    #again randomly every time you run the program

    #change the numbers to what you want to generate
    #2 models per category
    #3 vps for each model(each vp will be used for only once)
    # print("sampling data.....")
    # result_dict = random_sample_objs_and_vps(10, 2)
    # if not os.path.exists(g_temp):
    #     os.mkdir(g_temp)

    #I need to serialize path ad vp since subprocess.run can't pass 
    #python object
    # with open(os.path.join(g_temp, g_result_dict), 'wb') as f:
    #     pickle.dump(result_dict, f)

    #render fmo
    command = [g_blender_excutable_path, '--background', '-noaudio', '--python', 'render_fmo.py', '--', sys.argv[1], sys.argv[2]]
    subprocess.run(command)

    #render rgb
    # command = [g_blender_excutable_path, '--background', '-noaudio', '--python', 'render_rgb.py']
    # subprocess.run(command)

    #render depth
    # command = [g_blender_excutable_path, '--background', '-noaudio', '--python', 'render_depth.py']
    # subprocess.run(command)

    #write pose
    # command = [g_blender_excutable_path, '--background', '--python', 'render_pose.py']
    # subprocess.run(command)

