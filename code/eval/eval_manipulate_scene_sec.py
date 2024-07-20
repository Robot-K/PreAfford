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
import h5py
# check if there is a segfault
import faulthandler
faulthandler.enable()

parser = ArgumentParser()
parser.add_argument('--file', type=str, default='xxx')
parser.add_argument('--file_id', type=int, default=0)
parser.add_argument('--random_seed', type=int, default=None)
parser.add_argument('--density', type=float, default=5.0)
parser.add_argument('--damping', type=float, default=100.)
parser.add_argument('--start_dist', type=float, default=0.4) # 0.12
parser.add_argument('--maneuver_dist', type=float, default=0.5) 
parser.add_argument('--displacement', type=float, default=0.50)
parser.add_argument('--move_steps', type=int, default=2000)
parser.add_argument('--maneuver_steps', type=int, default=1500)
parser.add_argument('--wait_steps', type=int, default=1000)
parser.add_argument('--threshold', type=int, default=40)
parser.add_argument('--threshold_t', type=int, default=0.15)
parser.add_argument('--makedir', action='store_true', default=False)
parser.add_argument('--save_data', action='store_true', default=True)
parser.add_argument('--dir', type=str, default='xxx')
parser.add_argument('--no_gui', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--multiple_check', action='store_true', default=False)
args = parser.parse_args()

try:
    with open(args.file, 'r') as fin:
        result_data = json.load(fin)
except Exception:
    print('fail to load %s' % args.file)
    exit(3)

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

shape_id = result_data['shape_id']
category = result_data['category']
trial_id = args.file_id
scene = result_data['scene']
out_dir = args.dir

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

def save_succ_data(out_info, cam_XYZA_list, gt_link_mask, gif_imgs, repeat_id):
    if args.multiple_check:
        repeat_id = 0
    utils.save_data(os.path.join(out_dir, 'succ_files'), trial_id + repeat_id, out_info, cam_XYZA_list, gt_link_mask)
    imageio.mimsave(os.path.join(out_dir, 'succ_gif', '%d_%s_%s_%s.gif' % (trial_id + repeat_id, scene, category, shape_id)), gif_imgs)

def save_fail_data(out_info, cam_XYZA_list, gt_link_mask, gif_imgs, repeat_id):
    if args.multiple_check:
        repeat_id = 0
    if args.save_data:
        utils.save_data(os.path.join(out_dir, 'fail_files'), trial_id + repeat_id, out_info, cam_XYZA_list, gt_link_mask)
        if gif_imgs is not None:
            imageio.mimsave(os.path.join(out_dir, 'fail_gif', '%d_%s_%s_%s.gif' % (trial_id + repeat_id, scene, category, shape_id)), gif_imgs)

def save_invalid_data(out_info, cam_XYZA_list, gt_link_mask, fimg, repeat_id):
    if args.multiple_check:
        repeat_id = 0
    if args.save_data:
        utils.save_data(os.path.join(out_dir, 'invalid_files'), trial_id + repeat_id, out_info, cam_XYZA_list, gt_link_mask)
        fimg.save(os.path.join(out_dir, 'invalid_gif', '%d_%s_%s_%s.png' % (trial_id + repeat_id, scene, category, shape_id)))

def save_data(out_info, cam_XYZA_list, gt_link_mask, fimg):
    utils.save_data(os.path.join(out_dir, 'total_files'), trial_id, out_info, cam_XYZA_list, gt_link_mask)
    fimg.save(os.path.join(out_dir, 'total_gif', '%d_%s_%s_%s.png' % (trial_id, scene, category, shape_id)))

out_info = dict()
gif_imgs = []
fimg = None
success = False
out_info['scene'] = scene
# setup env
print("creating env")
start_dist = 0.34 # for debug
if scene == 'table':
    have_table = True
    # table
    size_x = 1.5
    size_y = 1.5
    thinkness = 0.1
    height = 1.0
    # print(height)
    # have greater probablity for larger values
    x_offset = (0.1+np.random.uniform()**0.7*size_x*0.4)*(1 if np.random.randint(2) == 0 else -1)
    y_offset = (0.1+np.random.uniform()**0.7*size_x*0.4)*(1 if np.random.randint(2) == 0 else -1)
    z_offset = 0.06 + height

    env = Env(show_gui=(not args.no_gui), set_ground=True, object_position_offset=[x_offset, y_offset, z_offset])
    env.create_table(Pose([0, 0, 0]), size_x, size_y, height, thinkness, (0.8, 0.6, 0.4), 1.0, 0.01, 1000, 'table')
    
    check_strict = False
    obj_random_orientation = 0.2

if scene == 'wall':
    have_table = False

    x_offset = 0.05
    gravity = [0., 0, -9.81]

    z_offset = 0.06
    y_offset = 0.
    
    env = Env(show_gui=(not args.no_gui), set_ground=True, object_position_offset=[x_offset, y_offset, z_offset], gravity=gravity)
    env.create_box(Pose([-0.5, 0, 0.5], [1,0,0,0]), [0.5,1.0,0.5], (0.8, 0.6, 0.4), friction=0.3, restitution=0.01, density=1000, name='wall')
    check_strict = False

    obj_random_orientation = 0.2


if scene == 'groove':
    have_table = False

    x_offset = (-1+2*np.random.uniform())*0.075
    z_offset = 0.1

    y_offset = 0.
    
    env = Env(show_gui=(not args.no_gui), set_ground=True, object_position_offset=[x_offset, y_offset, z_offset])
    groove_urdf = r'../../Shape_data/env_assets/groove_wide/urdf/groove_wide.urdf'
    # env.create_groove(groove_urdf, Pose([0,0,0.1], [1,0,0,0]))
    env.create_groove(gap=0.15)
    check_strict = False

    print("camera created")

    obj_random_orientation = 0.2

if scene == 'slope':
    have_table = False
    
    x_offset = np.random.uniform()*0.5
    z_offset = 0.1
    y_offset = 0.
    
    env = Env(show_gui=(not args.no_gui), set_ground=True, object_position_offset=[x_offset, y_offset, z_offset])
    groove_urdf = r'../../Shape_data/env_assets/slope/urdf/slope.urdf'
    env.create_slope(groove_urdf, Pose([0,0,0], [0.5**0.5, -0.5**0.5, 0, 0]), scale=3.0)
    check_strict = False
    print("camera created")

    obj_random_orientation = 0.2

if scene == 'multiple':
    have_table = True
    
    x_offset = np.random.uniform(-0.6, 0.6)
    z_offset = 1.1
    y_offset = 0.
    
    env = Env(show_gui=(not args.no_gui), set_ground=True, object_position_offset=[x_offset, y_offset, z_offset])
    env.create_multiple_scenes()
    dist = np.random.uniform(3., 3.5)
    check_strict = False

    obj_random_orientation = 0.2

# dist=5
theta = result_data['camera_metadata']['theta']
phi = result_data['camera_metadata']['phi']
dist = result_data['camera_metadata']['dist']
mat44 = result_data['camera_metadata']['mat44']
cam2cambase = result_data['camera_metadata']['cam2cambase']
cam = Camera(env, theta=theta, phi=phi, dist=dist, have_table=have_table, table_height=1.0, pos_rand_ratio=0., object_centered=True, mat44=mat44, cam2cambase=cam2cambase)
print("camera created")

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
    target_part_state = 'closed'
object_material = env.get_material(1.0, 1.0, 0.01) # static friction, dynamic friction, restitution
try:
    if category in mini_categories:
        joint_angles = env.load_object(object_urdf_fn, object_material, state=target_part_state, scale=0.15, density=args.density, damping=args.damping, stiffness=5., rand_orientation=obj_random_orientation, lieDown=Liedown, down=Down)
    elif category in smaller_categories:
        joint_angles = env.load_object(object_urdf_fn, object_material, state=target_part_state, scale=0.28, density=args.density, damping=args.damping, stiffness=5., rand_orientation=obj_random_orientation, lieDown=Liedown, down=Down)
    elif category in medium_categories:
        joint_angles = env.load_object(object_urdf_fn, object_material, state=target_part_state, scale=0.75, density=args.density, damping=args.damping, stiffness=5., rand_orientation=obj_random_orientation, lieDown=Liedown, down=Down)
    elif category in specific_categories:
        joint_angles = env.load_object(object_urdf_fn, object_material, state=target_part_state, scale=size_dict[shape_id], density=args.density, damping=args.damping, stiffness=5., rand_orientation=obj_random_orientation, lieDown=Liedown, down=Down)
    else:
        joint_angles = env.load_object(object_urdf_fn, object_material, state=target_part_state, scale=1.0, density=args.density, damping=args.damping, stiffness=5., rand_orientation=obj_random_orientation, lieDown=Liedown, down=Down)

except Exception:
    print('error while load object')
    for a in env.scene.get_all_actors():
        env.scene.remove_actor(a)
    env.close()
    exit(100)

# load inferrence results
try:
    with open(os.path.join(out_dir, 'total_files', 'inferred', f'result_{args.file_id}.json'), 'r') as fin:
        inferred_data = json.load(fin)
except Exception:
    print('fail to load inferred data of %s' % args.file)
    exit(4)

position1s = inferred_data['position1s']
# print(position1s)
dir1s = inferred_data['dir1s']
# print('dir1s = ', dir1s)
try:
    with h5py.File(os.path.join(out_dir, 'total_files', 'cambase', '_XYZA_boxed_%d.h5' % args.file_id), 'r') as fin:
        # print(pc.shape)
        pc_center = fin['pc_center'][:].astype(np.float32)
except Exception:
    print('fail to load %s' % os.path.join(out_dir, 'total_files', 'cambase', '_XYZA_boxed_%d.h5' % args.file_id))
    exit(5)

env.contact_error = None

num_pair = len(position1s)
for trial in range(num_pair):
    print('trial = ', trial)
    obj_root_pose_p = result_data['obj_root_pose_p']
    obj_root_pose_q = result_data['obj_root_pose_q']
    obj_root_pose = Pose(obj_root_pose_p, obj_root_pose_q)
    still_joint_angles = result_data['still_joint_angles']

    env.object.set_pose(obj_root_pose)
    env.object.set_root_velocity(np.zeros(6))
    env.object.set_qpos(still_joint_angles)
    env.object.set_qvel(np.zeros_like(still_joint_angles))
    env.contact_error = None
    env.check_contact = False
    still_timesteps = utils.wait_for_object_still(env, have_table=have_table)
    after_still_joint_angles = env.object.get_qpos()
    after_still_root_pose = env.object.get_pose()
    if len(after_still_joint_angles) > 0:
        max_q_displacement = np.max(np.abs(after_still_joint_angles - still_joint_angles))
    else:
        max_q_displacement = 0
    max_root_displacement = np.max(np.abs(after_still_root_pose.p - obj_root_pose.p))
    if max_q_displacement > 0.02 or max_root_displacement > 0.02:
        print("object not still")
        env.scene.remove_articulation(env.object)
        for a in env.scene.get_all_actors():
            env.scene.remove_actor(a)
        env.close()
        exit(101)
    env.render()
    # wait for the object's still
    # qp_monitor = monitor.MonitorApp(window_title='qp', interval=10)

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

    if env.target_object_part_joint_type != target_joint_type:
        pre_target_link_mat44 = env.get_object_root_pose().to_transformation_matrix()  # local2world
        prev_origin_world_xyz1 = pre_target_link_mat44 @ np.array([0, 0, 0, 1])
        prev_origin_world = prev_origin_world_xyz1[:3]
        obj_pose = env.get_object_root_pose()
        print("no_fixed_part")

    env.render()
    rgb_pose, _ = cam.get_observation()
    fimg = (rgb_pose * 255).astype(np.uint8)
    fimg = Image.fromarray(fimg)

    env.render()
    
    # inference for manipulation
    # print(np.array(position1s[trial]), np.array(dir1s[trial]))
    position_world1 = np.array(position1s[trial]).reshape(3)
    up_world1, forward_world1 = np.array(dir1s[trial][0]).reshape(3), np.array(dir1s[trial][1]).reshape(3)
    # cambase_batch = [position1.reshape(3), up1.reshape(3), forward1.reshape(3)]
    # is_pc = [True, False, False]
    # camera_batch = utils.batch_coordinate_transform(cambase_batch, is_pc, transform_type='cambase2cam', cam2cambase=cam.cam2cambase, pc_center=pc_center)
    # print('cam_batch', camera_batch)
    # world_batch = utils.batch_coordinate_transform(camera_batch, is_pc, transform_type='cam2world', mat44=cam.mat44, pc_center=pc_center)
    # print('world_batch', world_batch)
    # position_world1, up_world1, forward_world1 = world_batch
    print('position_world', position_world1)
    pre_pose, pre_rotmat, start_pose, start_rotmat = utils.get_rotmat(cam, position_world1, up_world1, forward_world1, number='1', out_info=out_info, start_dist=args.maneuver_dist, final_dist=args.start_dist)
    print(pre_pose, pre_rotmat, start_pose, start_rotmat)
    final_rotmat = start_rotmat.copy()
    final_rotmat[2, 3] += args.displacement
    
    # print("up = ", up, "forward = ", forward)
    # print("pre_pose = ", pre_pose, "pre_rotmat = ", pre_rotmat)

    # setup robot
    robot_urdf_fn = './airbot/urdf_obj/AIRBOT_hand.urdf'
    robot_material = env.get_material(4., 4., 0.01)
    robot_scale = 1.4

    robot = Robot(env, robot_urdf_fn, robot_material, open_gripper=True, scale=robot_scale)
    # qf_monitor = monitor.MonitorApp(window_title='qf', interval=10)
    qf_monitor=None

    # activate initial contact checking
    robot.robot.set_root_pose(pre_pose)
    env.start_checking_contact(robot.hand_actor_id, robot.gripper_actor_ids, strict=check_strict)
    print("start checking contact")
    # save img
    env.render()
    rgb_pose, _ = cam.get_observation()
    fimg = (rgb_pose * 255).astype(np.uint8)
    fimg = Image.fromarray(fimg)
    # save the fimg
    # fimg.save(f"./fimg{contact_trial}.png")
    # gif_imgs.extend(fimg)
    # print the caught exception
    env.step()

    if env.contact_error:
        if not args.multiple_check or trial == num_pair - 1:
            out_info['random_seed'] = args.random_seed
            out_info['target_object_part_joint_type'] = str(env.target_object_part_joint_type)
            out_info['camera_metadata'] = cam.get_metadata_json()
            out_info['object_state'] = target_part_state
            out_info['joint_angles'] = joint_angles
            out_info['joint_angles_lower'] = env.joint_angles_lower
            out_info['joint_angles_upper'] = env.joint_angles_upper
            out_info['shape_id'] = shape_id
            out_info['category'] = category
            out_info['scene'] = scene
            out_info['target_link_mat44'] = pre_target_link_mat44.tolist()
            out_info['prev_origin_world'] = prev_origin_world.tolist()
            out_info['obj_pose_p'] = obj_pose.p.tolist()
            out_info['obj_pose_q'] = obj_pose.q.tolist()
            out_info['success'] = 'False'
            out_info['result'] = 'INVALID'
            save_invalid_data(out_info, cam_XYZA_list, gt_all_link_mask, fimg, trial)
            env.scene.remove_articulation(robot.robot)
        if trial == num_pair - 1:
            env.scene.remove_articulation(env.object)
            for a in env.scene.get_all_actors():
                env.scene.remove_actor(a)
            env.close()
            exit(2)
        env.scene.remove_articulation(robot.robot)
        continue

    out_info['random_seed'] = args.random_seed
    out_info['target_object_part_joint_type'] = str(env.target_object_part_joint_type)
    out_info['camera_metadata'] = cam.get_metadata_json()
    out_info['object_state'] = target_part_state
    out_info['joint_angles'] = joint_angles
    out_info['joint_angles_lower'] = env.joint_angles_lower
    out_info['joint_angles_upper'] = env.joint_angles_upper
    out_info['shape_id'] = shape_id
    out_info['category'] = category
    out_info['scene'] = scene
    out_info['target_link_mat44'] = pre_target_link_mat44.tolist()
    out_info['prev_origin_world'] = prev_origin_world.tolist()
    out_info['obj_pose_p'] = obj_pose.p.tolist()
    out_info['obj_pose_q'] = obj_pose.q.tolist()
    out_info['success'] = 'False'
    # out_info['success_bd'] = 'False'
、‘；    out_info['result'] = 'VALID'

    con = utils.check_con(env, cam, cam_XYZA, scene)
    # if not con and category in Thin_categories:
    #     if not args.multiple_check or trial == num_pair - 1:
    #         save_fail_data(out_info, cam_XYZA_list, gt_all_link_mask, None, trial)
    #     env.scene.remove_articulation(robot.robot)
    #     if trial == num_pair - 1:
    #         env.scene.remove_articulation(env.object)
    #         for a in env.scene.get_all_actors():
    #             env.scene.remove_actor(a)
    #         env.close()
    #         exit(1)
    #     continue

    start_target_link_mat44 = env.get_target_part_pose().to_transformation_matrix()  # local2world
    start_origin_world_xyz1 = start_target_link_mat44 @ np.array([0, 0, 0, 1])
    start_origin_world = start_origin_world_xyz1[:3]
    obj_pose = env.get_target_part_pose()

    if env.target_object_part_joint_type != target_joint_type:
        start_target_link_mat44 = env.get_object_root_pose().to_transformation_matrix()  # local2world
        start_origin_world_xyz1 = start_target_link_mat44 @ np.array([0, 0, 0, 1])
        start_origin_world = start_origin_world_xyz1[:3]
        obj_pose = env.get_object_root_pose()

    # time.sleep(5)
    env.raise_contact_error = True
    env.check_contact_strict = False

    try:
        print("start maneuver")
        print(start_rotmat)
        imgs = utils.gripper_move_to_target_pose(robot, start_rotmat, num_steps=args.maneuver_steps, cam=cam, vis_gif=True, vis_gif_interval=100, monitor = qf_monitor)
        
        gif_imgs.extend(imgs)
        print("start closing gripper")
        robot.close_gripper()
        imgs = utils.gripper_wait_n_steps(robot, n=args.wait_steps, cam=cam, vis_gif=True, vis_gif_interval=100, monitor = qf_monitor)
        gif_imgs.extend(imgs)
        # print(len(imgs))
        print("start picking up")
        imgs = utils.gripper_move_to_target_pose(robot, final_rotmat, num_steps=args.move_steps, cam=cam, vis_gif=True, vis_gif_interval=100, monitor = qf_monitor)
        gif_imgs.extend(imgs)
        # print(len(imgs))
        print("start waiting")
        imgs = utils.gripper_wait_n_steps(robot, n=args.wait_steps, cam=cam, vis_gif=True, vis_gif_interval=100, monitor = qf_monitor)
        gif_imgs.extend(imgs)
        env.end_checking_contact(robot.hand_actor_id, robot.gripper_actor_ids, False)
        print("end checking contact")
        print(len(imgs))

    except Exception:
        if not args.multiple_check or trial == num_pair - 1:
            print('contact error')
            out_info['result'] = 'INVALID'
            save_invalid_data(out_info, cam_XYZA_list, gt_all_link_mask, fimg, trial)
        env.scene.remove_articulation(robot.robot)
        if trial == num_pair - 1:
            env.scene.remove_articulation(env.object)
            for a in env.scene.get_all_actors():
                env.scene.remove_actor(a)
            env.close()
            exit(2)
        continue

    ''' check success '''

    next_obj_pose = env.get_target_part_pose()
    if env.target_object_part_joint_type != target_joint_type:
        next_obj_pose = env.get_object_root_pose()

    target_part_trans = next_obj_pose.to_transformation_matrix()  # world coordinate -> target part transformation matrix 4*4 SE3

    transition = np.linalg.inv(target_part_trans) @ pre_target_link_mat44
    alpha, beta, gamma = utils.rotationMatrixToEulerAngles(transition)  # eulerAngles(trans) = eulerAngles(prev_mat44) - eulerAngle(then_mat44) # in degree
    out_info['start_origin_world'] = start_origin_world.tolist()
    out_info['target_part_trans'] = target_part_trans.tolist()
    out_info['transition'] = transition.tolist()
    out_info['alpha'] = alpha.tolist()
    out_info['beta'] = beta.tolist()
    out_info['gamma'] = gamma.tolist()
    out_info['next_obj_pose_p'] = next_obj_pose.p.tolist()
    out_info['next_obj_pose_q'] = next_obj_pose.q.tolist()

    # calculate displacement
    next_origin_world_xyz1 = target_part_trans @ np.array([0, 0, 0, 1])
    next_origin_world = next_origin_world_xyz1[:3]
    trajectory = next_origin_world - prev_origin_world
    # print('before check success')
    success, traj_x, traj_y, traj_z = utils.check_success(trajectory, alpha, beta, gamma, threshold=args.threshold, threshold_t=args.threshold_t, displacement = args.displacement)
    out_info['success'] = 'True' if success else 'False'
    out_info['trajectory'] = trajectory.tolist()
    out_info['traj_x'] = traj_x.tolist()
    out_info['traj_y'] = traj_y.tolist()
    out_info['traj_z'] = traj_z.tolist()
    # print('after check success')

    env.scene.remove_articulation(robot.robot)
    if success:
        save_succ_data(out_info, cam_XYZA_list, gt_all_link_mask, gif_imgs, trial)
        if args.multiple_check:
            env.scene.remove_articulation(env.object)
            for a in env.scene.get_all_actors():
                env.scene.remove_actor(a)
            env.close()
            print('exit in 0')
            exit(0)
    else:
        if not args.multiple_check or trial == num_pair - 1:
            save_fail_data(out_info, cam_XYZA_list, gt_all_link_mask, gif_imgs, trial)
        if trial == num_pair - 1:
            env.scene.remove_articulation(env.object)
            for a in env.scene.get_all_actors():
                env.scene.remove_actor(a)
            env.close()
            print('exit in 1')
            exit(1)

env.scene.remove_articulation(env.object)
for a in env.scene.get_all_actors():
    env.scene.remove_actor(a)
env.close()
exit(103)