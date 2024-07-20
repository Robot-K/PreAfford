import os
import numpy as np
from PIL import Image
import utils
from argparse import ArgumentParser
from sapien.core import Pose, ArticulationJointType
from env import Env, ContactError, SVDError
from camera import Camera
from airbot.airbot import Robot
import random
import imageio 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import monitor
import subprocess
import torch
import json
# check if there is a segfault
import faulthandler
faulthandler.enable()

parser = ArgumentParser()
parser.add_argument('--trial_id', type=int)
parser.add_argument('--shape_id', type=str)
parser.add_argument('--category', type=str)
parser.add_argument('--out_dir', type=str)
parser.add_argument('--random_seed', type=int, default=None)
parser.add_argument('--scene', type=str, default='table')
parser.add_argument('--density', type=float, default=5.0)
parser.add_argument('--damping', type=float, default=100.)
parser.add_argument('--target_part_state', type=str, default='random-middle')
parser.add_argument('--start_dist', type=float, default=0.4) # 0.12
parser.add_argument('--maneuver_dist', type=float, default=0.5) 
parser.add_argument('--displacement', type=float, default=0.50)
parser.add_argument('--move_steps', type=int, default=2000)
parser.add_argument('--maneuver_steps', type=int, default=1500)
parser.add_argument('--wait_steps', type=int, default=1000)
parser.add_argument('--threshold', type=int, default=40)
parser.add_argument('--threshold_t', type=int, default=0.15)
parser.add_argument('--con', action='store_true', default=False)
parser.add_argument('--makedir', action='store_true', default=False)
parser.add_argument('--save_data', action='store_true', default=False)
parser.add_argument('--no_gui', action='store_true', default=False, help='no_gui [default: False]')

parser.add_argument('--z_dim', type=int, default=10)
parser.add_argument('--feat_dim', type=int, default=160)
parser.add_argument('--task_feat_dim', type=int, default=32)
parser.add_argument('--cp_feat_dim', type=int, default=32)
parser.add_argument('--dir_feat_dim', type=int, default=32)
parser.add_argument('--num_point_per_shape', type=int, default=8192)

parser.add_argument('--use_4th_feature', action='store_true', default=False)
parser.add_argument('--aff1_version', type=str, default=None)
parser.add_argument('--aff1_path', type=str, default=None)
parser.add_argument('--aff1_eval_epoch', type=str, default=None)
parser.add_argument('--actor1_version', type=str, default=None, help='model def file')
parser.add_argument('--actor1_path', type=str)
parser.add_argument('--actor1_eval_epoch', type=str)
parser.add_argument('--critic1_version', type=str, default=None, help='model def file')
parser.add_argument('--critic1_path', type=str)
parser.add_argument('--critic1_eval_epoch', type=str)

parser.add_argument('--CA_path', type=str, default=None)
parser.add_argument('--CA_eval_epoch', type=str, default=None)
parser.add_argument('--use_CA', action='store_true', default=False)

parser.add_argument('--aff_topk', type=float, default=0.1)
parser.add_argument('--critic_topk1', type=float, default=0.01)
parser.add_argument('--num_ctpt1', type=int, default=10)
parser.add_argument('--rv1', type=int, default=100)
parser.add_argument('--num_pair1', type=int, default=10)
parser.add_argument('--num_ctpts', type=int, default=10)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--cat_cnt_dict_path', type=str, default='xxx')
parser.add_argument('--scene_cnt_dict_path', type=str, default='xxx')
parser.add_argument('--multiple_check', action='store_true', default=False)

args = parser.parse_args()

def print_gpu():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.free', '--format=csv,nounits,noheader'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        if result.returncode != 0:
            print("Error running nvidia-smi")
            return

        gpu_memory = result.stdout.strip().split('\n')[0].split(',')
        if len(gpu_memory) >= 2:
            used_memory = gpu_memory[0].strip()
            free_memory = gpu_memory[1].strip()
            print(f"Used GPU Memory: {used_memory} MB")
            print(f"Free GPU Memory: {free_memory} MB")
        else:
            print("Error parsing nvidia-smi output")
    except Exception as e:
        print(f"An error occurred: {e}")

def write_json(cat_cnt_dict_path, scene_cnt_dict_path, selected_cat, scene, result):
    with open(cat_cnt_dict_path, 'r') as json_file:
        cat_cnt_dict = json.load(json_file)
    with open(scene_cnt_dict_path, 'r') as json_file:
        scene_cnt_dict = json.load(json_file)
    cat_cnt_dict[selected_cat][result] += 1
    scene_cnt_dict[scene][result] += 1
    if result != 'total':
        cat_cnt_dict[selected_cat]['total'] += 1
        scene_cnt_dict[scene]['total'] += 1
    with open(cat_cnt_dict_path, 'w') as json_file:
        json.dump(cat_cnt_dict, json_file)
    with open(scene_cnt_dict_path, 'w') as json_file:
        json.dump(scene_cnt_dict, json_file)

shape_id = args.shape_id
category = args.category
trial_id = args.trial_id
scene = args.scene
out_dir = args.out_dir
if args.random_seed is not None:
    np.random.seed(args.random_seed)

if args.makedir:
    if os.path.exists(out_dir):
        response = input('Out directory "%s" already exists, now covering it.' % out_dir)

    if not os.path.exists(out_dir):
        os.makedirs(os.path.join(out_dir, 'succ_gif'))
        os.makedirs(os.path.join(out_dir, 'fail_gif'))
        os.makedirs(os.path.join(out_dir, 'invalid_gif'))
        # os.makedirs(os.path.join(out_dir, 'tmp_succ_gif'))

        os.makedirs(os.path.join(out_dir, 'succ_files'))
        os.makedirs(os.path.join(out_dir, 'fail_files'))
        os.makedirs(os.path.join(out_dir, 'invalid_files'))
        # os.makedirs(os.path.join(out_dir, 'tmp_succ_files'))

def save_succ_data(out_info, cam_XYZA_list, gt_link_mask, gif_imgs, trial):
    utils.save_data(os.path.join(out_dir, 'succ_files'), trial_id, out_info, cam_XYZA_list, gt_link_mask, repeat_id=trial)
    imageio.mimsave(os.path.join(out_dir, 'succ_gif', '%d_%s_%s_%s_%d.gif' % (trial_id, scene, category, shape_id, trial)), gif_imgs)

def save_fail_data(out_info, cam_XYZA_list, gt_link_mask, gif_imgs, trial):
    if args.save_data:
        utils.save_data(os.path.join(out_dir, 'fail_files'), trial_id, out_info, cam_XYZA_list, gt_link_mask, repeat_id=trial)
        imageio.mimsave(os.path.join(out_dir, 'fail_gif', '%d_%s_%s_%s_%d.gif' % (trial_id, scene, category, shape_id, trial)), gif_imgs)

def save_invalid_data(out_info, cam_XYZA_list, gt_link_mask, fimg, trial):
    if args.save_data:
        utils.save_data(os.path.join(out_dir, 'invalid_files'), trial_id, out_info, cam_XYZA_list, gt_link_mask, repeat_id=trial)
        fimg.save(os.path.join(out_dir, 'invalid_gif', '%d_%s_%s_%s_%d.png' % (trial_id, scene, category, shape_id, trial)))

def save_data(out_info, cam_XYZA_list, gt_link_mask, fimg, trial):
    utils.save_data(os.path.join(out_dir, 'total_files'), trial_id, out_info, cam_XYZA_list, gt_link_mask)
    fimg.save(os.path.join(out_dir, 'total_gif', '%d_%s_%s_%s.png' % (trial_id, scene, category, shape_id)))

out_info = dict()
gif_imgs = []
fimg = None
success = False
out_info['scene'] = scene
# setup env
print("creating env")

if args.scene == 'table':
    have_table = True
    # table
    size_x = 1.5
    size_y = 1.5
    thinkness = 0.1
    height = 1.0
    # print(height)
    # have greater probablity for larger values
    if not args.con:
        x_offset = np.random.uniform()**0.7*size_x*0.3*(1 if np.random.randint(2) == 0 else -1)
        y_offset = np.random.uniform()**0.7*size_y*0.3*(1 if np.random.randint(2) == 0 else -1)
        z_offset = 0.06 + height
    if args.con:
        x_offset = (0.1+np.random.uniform()**0.7*size_x*0.4)*(1 if np.random.randint(2) == 0 else -1)
        y_offset = (0.1+np.random.uniform()**0.7*size_x*0.4)*(1 if np.random.randint(2) == 0 else -1)
        z_offset = 0.06 + height

    env = Env(show_gui=(not args.no_gui), set_ground=True, object_position_offset=[x_offset, y_offset, z_offset])
    env.create_table(Pose([0, 0, 0]), size_x, size_y, height, thinkness, (0.8, 0.6, 0.4), 1.0, 0.01, 1000, 'table')
    dist = np.random.uniform(3., 3.5)

    # dist=5
    cam = Camera(env, fixed_position=True, dist=dist, have_table=have_table, table_height=height, pos_rand_ratio=0.2, object_centered=True)
    print("camera created")
    check_strict = False

    obj_random_orientation = 0.2

if args.scene == 'wall':
    have_table = False
    if not args.con:
        x_offset = np.random.uniform()*0.6 + 0.3
        gravity = [0., 0, -9.81]

    if args.con:
        x_offset = 0.05
        gravity = [0., 0, -9.81]

    z_offset = 0.06
    y_offset = 0.
    
    env = Env(show_gui=(not args.no_gui), set_ground=True, object_position_offset=[x_offset, y_offset, z_offset], gravity=gravity)
    env.create_box(Pose([-0.5, 0, 0.5], [1,0,0,0]), [0.5,1.0,0.5], (0.8, 0.6, 0.4), friction=0.3, restitution=0.01, density=1000, name='wall')
    dist = np.random.uniform(3., 3.5)
    check_strict = False

    # dist=5
    cam = Camera(env, fixed_position=True, dist=dist, pos_rand_ratio=0.2, object_centered=True)
    print("camera created")

    obj_random_orientation = 0.2


if args.scene == 'groove':
    have_table = False
    if not args.con:
        x_offset = 0.3 + np.random.uniform()*0.6
    if args.con:
        x_offset = (-1+2*np.random.uniform())*0.075
    z_offset = 0.1

    y_offset = 0.
    
    env = Env(show_gui=(not args.no_gui), set_ground=True, object_position_offset=[x_offset, y_offset, z_offset])
    groove_urdf = r'../../Shape_data/env_assets/groove_wide/urdf/groove_wide.urdf'
    # env.create_groove(groove_urdf, Pose([0,0,0.1], [1,0,0,0]))
    env.create_groove(gap=0.15)
    dist = np.random.uniform(3., 3.5)
    check_strict = False

    # dist=5
    cam = Camera(env, fixed_position=True, dist=dist, pos_rand_ratio=0.2, object_centered=True)
    print("camera created")

    obj_random_orientation = 0.2

if args.scene == 'slope':
    have_table = False
    
    if not args.con:
        x_offset = np.random.uniform()*0.5 + 0.3
        z_offset = 0.06
    else:
        x_offset = np.random.uniform()*0.5
        z_offset = 0.1
    y_offset = 0.

    x_offset = 0.4
    
    env = Env(show_gui=(not args.no_gui), set_ground=True, object_position_offset=[x_offset, y_offset, z_offset])
    groove_urdf = r'../../Shape_data/env_assets/slope/urdf/slope.urdf'
    env.create_slope(groove_urdf, Pose([0,0,0], [0.5**0.5, -0.5**0.5, 0, 0]), scale=3.0)
    dist = np.random.uniform(3., 3.5)
    dist=3
    check_strict = False

    # dist=5
    cam = Camera(env, fixed_position=True, dist=dist, pos_rand_ratio=0., object_centered=True)
    print("camera created")

    obj_random_orientation = 1.0

if args.scene == 'multiple':
    have_table = True
    
    x_offset = np.random.uniform(-0.6, 0.6)
    z_offset = 1.1
    y_offset = np.random.uniform(-0.6, 0.6)
    
    env = Env(show_gui=(not args.no_gui), set_ground=True, object_position_offset=[x_offset, y_offset, z_offset])
    env.create_multiple_scenes()
    dist = np.random.uniform(3., 3.5)
    check_strict = False
    dist=3

    # dist=5
    cam = Camera(env, fixed_position=True, dist=dist, pos_rand_ratio=0., object_centered=False, have_table=have_table, if_front=True)
    print("camera created")

    obj_random_orientation = 0.2

# print(cam.pos, cam.theta, cam.phi)
if not args.no_gui:
    env.set_controller_camera_pose(cam.pos[0], cam.pos[1], cam.pos[2], np.pi + cam.theta, -cam.phi)

specific_categories = ['Phone']
size_dict = {'22001': 0.00035, '22002': 0.0015, '22003': 0.0005, '22004': 0.013}
mini_categories = ['Switch', 'Pen', 'USB', 'Bowl', 'Remote', 'Bowl_lie']
smaller_categories = ['Box', 'Display', 'Bucket', 'Eyeglasses', 'Faucet', 'Kettle', 'Keyboard', 'KitchenPot', 'Laptop', 'Microwave', 'Oven', 'Pliers', 'Scissors', 'Toaster', 'Window', 'Basket', 'Keyboard2', 'Cap']
medium_categories = ['Dishwasher', 'Printer', 'Refrigerator', 'Safe', 'StorageFurniture', 'Basket', 'Chair2']
Large_categories = ['Cart', 'Chair', 'Door', 'FoldingChair', 'Table', 'TrashCan', 'WashingMachine', 'Bench', 'Sofa']

PartNet_categories = ['Phone', 'Box', 'Bucket', 'Cart', 'Chair', 'Dishwasher', 'Display', 'Door', 'Eyeglasses', 'Faucet', 'FoldingChair', 'Kettle', 'Keyboard', 'KitchenPot', 'Laptop', 'Microwave', 'Oven', 'Pen', 'Pliers', 'Printer', 'Refrigerator', 'Remote', 'Safe', 'Scissors', 'StorageFurniture', 'Switch', 'Table', 'Toaster', 'TrashCan', 'USB', 'WashingMachine', 'Window']
ShapeNet_categories = ['Basket', 'Bench', 'Bowl', 'Bowl_lie', 'Chair2', 'Keyboard2', 'Sofa', 'Cap']

Thin_categories = ['Switch', 'Laptop', 'Remote', 'Scissors', 'Window', 'Keyboard2', 'Pen', 'USB', 'Bowl_lie', 'Cap', 'Phone']
Pickable_categories = ['Box', 'Bucket', 'Display', 'Eyeglasses', 'Faucet', 'Kettle', 'KitchenPot', 'Pliers', 'Basket', 'Bowl']
Unpickable_categories = ['Cart', 'Chair', 'Dishwasher', 'Door', 'FoldingChair', 'Microwave', 'Oven', 'Printer', 'Refrigerator', 'Safe', 'StorageFurniture', 'Table', 'Toaster', 'TrashCan', 'WashingMachine', 'Bench', 'Chair2', 'Sofa']
Liedown_categories = ['Switch', 'Remote', 'Window', 'Pen', 'Phone']
Down_categories = ['Bowl_lie']

Middle_state_categories = ['Display']

if category in PartNet_categories:
    object_urdf_fn = '../../Shape_data/dataset/%s/mobility.urdf' % str(shape_id)
elif category in ShapeNet_categories: # PartNet-Mobility Dataset
    object_urdf_fn = '../../Shape_data/dataset2/%s/mobility_vhacd.urdf' % str(shape_id)
else:
    print('category not found')
    exit(4)
Liedown = False
Down = False
if category in Liedown_categories:
    Liedown = True
if category in Down_categories:
    Down = True

if category in Middle_state_categories:
    target_part_state = 'middle'
else:
    target_part_state = args.target_part_state
object_material = env.get_material(1.0, 1.0, 0.01) # static friction, dynamic friction, restitution
try:
    if category in mini_categories:
        joint_angles = env.load_object(object_urdf_fn, object_material, state=target_part_state, scale=0.15, density=args.density, damping=args.damping, stiffness=5., rand_orientation=obj_random_orientation, lieDown=Liedown, down=Down)
    elif category in smaller_categories:
        joint_angles = env.load_object(object_urdf_fn, object_material, state=target_part_state, scale=0.28, density=args.density, damping=args.damping, stiffness=5., rand_orientation=obj_random_orientation, lieDown=Liedown, down=Down)
    elif category in medium_categories:
        joint_angles = env.load_object(object_urdf_fn, object_material, state=target_part_state, scale=0.75, density=args.density, damping=args.damping, stiffness=5., rand_orientation=obj_random_orientation, lieDown=Liedown, down=Down)
    elif category in specific_categories:
        joint_angles = env.load_object(object_urdf_fn, object_material, state=target_part_state, scale=size_dict[args.shape_id], density=args.density, damping=args.damping, stiffness=5., rand_orientation=obj_random_orientation, lieDown=Liedown, down=Down)
    else:
        joint_angles = env.load_object(object_urdf_fn, object_material, state=target_part_state, scale=1.0, density=args.density, damping=args.damping, stiffness=5., rand_orientation=obj_random_orientation, lieDown=Liedown, down=Down)

except Exception:
    print('error while load object')
    for a in env.scene.get_all_actors():
        env.scene.remove_actor(a)
    env.close()
    exit(100)

# wait for the object's still
# qp_monitor = monitor.MonitorApp(window_title='qp', interval=10)
qp_monitor = None
still_timesteps = utils.wait_for_object_still(env, have_table=have_table, monitor=qp_monitor)

# save the gif
# still_imgs = []
# still_timesteps, imgs = utils.wait_for_object_still(env, cam=cam, visu=True)
# still_imgs.extend(imgs)
# if still_timesteps < 5000:
#     imageio.mimsave(os.path.join(args.out_dir, args.out_folder, '%d_%d_%s_%s.gif' % (trial, idx_process, selected_cat, shape_id)), still_imgs)

if still_timesteps < 200:
    print('Object Not Still!')
    env.scene.remove_articulation(env.object)
    for a in env.scene.get_all_actors():
        env.scene.remove_actor(a)
    env.close()
    # write category, shape_id
    with open('./wrong_obj.txt', 'a') as fout:
        fout.write('%s %s\n' % (category, shape_id))
    exit(101)

### use the GT vision
rgb, depth = cam.get_observation()
cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = cam.compute_camera_XYZA(depth) # return the point cloud between near and far
object_movable_link_ids = env.movable_link_ids
object_all_link_ids = env.all_link_ids
cam_XYZA = cam.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, depth.shape[0], depth.shape[1], object_all_link_ids) # [depth.shape[0], depth.shape[1], 4]

cam_XYZA_list = [cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, cam_XYZA]
gt_movable_link_mask = cam.get_movable_link_mask(object_movable_link_ids)  # (448, 448), 0(unmovable) - id(movable)
gt_all_link_mask = cam.get_movable_link_mask(object_all_link_ids)  # (448, 448), 0(unmovable) - id(all)

# sample a pixel on target part
xs, ys = np.where(gt_all_link_mask > 0)
if len(xs) == 0:
    env.scene.remove_articulation(env.object)
    env.close()
    print("no_target_point")
    exit(6)
idx = np.random.randint(len(xs))
x, y = xs[idx], ys[idx]

target_part_id = object_all_link_ids[gt_all_link_mask[x, y] - 1]
env.set_target_object_part_actor_id2(target_part_id)  # for get_target_part_pose
target_joint_type = ArticulationJointType.FIX
tot_trial = 0
while tot_trial < 50 and (env.target_object_part_joint_type != target_joint_type):
    idx = np.random.randint(len(xs))
    x, y = xs[idx], ys[idx]
    target_part_id = object_all_link_ids[gt_all_link_mask[x, y] - 1]
    env.set_target_object_part_actor_id2(target_part_id)
    tot_trial += 1

gt_target_link_mask = cam.get_movable_link_mask([target_part_id])

# calculate position world = trans @ local
pre_target_link_mat44 = env.get_target_part_pose().to_transformation_matrix()  # local2world
prev_origin_world_xyz1 = pre_target_link_mat44 @ np.array([0, 0, 0, 1])
prev_origin_world = prev_origin_world_xyz1[:3]
obj_pose = env.get_target_part_pose()
obj_root_pose = env.object.get_pose()
still_joint_angles = env.object.get_qpos()
print('object pose = ', obj_pose.p)

if env.target_object_part_joint_type != target_joint_type:
    # env.scene.remove_articulation(env.object)
    # for a in env.scene.get_all_actors():
    #     env.scene.remove_actor(a)
    # env.close()
    pre_target_link_mat44 = env.get_object_root_pose().to_transformation_matrix()  # local2world
    prev_origin_world_xyz1 = pre_target_link_mat44 @ np.array([0, 0, 0, 1])
    prev_origin_world = prev_origin_world_xyz1[:3]
    obj_pose = env.get_object_root_pose()
    print("no_fixed_part")
    # exit(7)

env.render()
rgb_pose, _ = cam.get_observation()
fimg = (rgb_pose * 255).astype(np.uint8)
fimg = Image.fromarray(fimg)
out_info['random_seed'] = args.random_seed
out_info['camera_metadata'] = cam.get_metadata_json()
out_info['object_state'] = args.target_part_state
out_info['still_joint_angles'] = still_joint_angles.tolist()
out_info['joint_angles_lower'] = env.joint_angles_lower
out_info['joint_angles_upper'] = env.joint_angles_upper
out_info['shape_id'] = shape_id
out_info['category'] = category
out_info['scene'] = scene
out_info['obj_root_pose_p'] = obj_root_pose.p.tolist()
out_info['obj_root_pose_q'] = obj_root_pose.q.tolist()
print(out_info)
con = utils.check_con(env, cam, cam_XYZA, args.scene)
print("con = ", con)
if args.con and not con and args.scene != 'multiple':
    env.scene.remove_articulation(env.object)
    for a in env.scene.get_all_actors():
        env.scene.remove_actor(a)
    env.close()
    print("no_con while con")
    exit(102)

if not args.con and con and args.scene != 'multiple':
    env.scene.remove_articulation(env.object)
    for a in env.scene.get_all_actors():
        env.scene.remove_actor(a)
    env.close()
    print("con while should not con")
    exit(103)

save_data(out_info, cam_XYZA_list, gt_all_link_mask, fimg, trial_id)
write_json(args.cat_cnt_dict_path, args.scene_cnt_dict_path, category, scene, 'total')
env.scene.remove_articulation(env.object)
for a in env.scene.get_all_actors():
    env.scene.remove_actor(a)
env.close()
exit(0)
