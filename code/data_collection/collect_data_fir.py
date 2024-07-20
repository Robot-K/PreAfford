# airbot modified damping
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
parser.add_argument('--move_steps', type=int, default=2000)
parser.add_argument('--maneuver_steps', type=int, default=1500)
parser.add_argument('--wait_steps', type=int, default=1000)
parser.add_argument('--threshold', type=int, default=20)
parser.add_argument('--threshold_t', type=int, default=0.1)
parser.add_argument('--makedir', action='store_true', default=False)
parser.add_argument('--save_data', action='store_true', default=False)
parser.add_argument('--use_edge', action='store_true', default=False)
parser.add_argument('--if_given_direction', action='store_true', default=False)
parser.add_argument('--no_gui', action='store_true', default=False, help='no_gui [default: False]')

args = parser.parse_args()

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

def save_succ_data(out_info, cam_XYZA_list, gt_link_mask, cam_XYZA_list_after, gt_link_mask_after, gif_imgs):
    utils.save_data(os.path.join(out_dir, 'succ_files'), trial_id, out_info, cam_XYZA_list, gt_link_mask, gt_link_mask_after=gt_link_mask_after, final_cam_XYZA_list=cam_XYZA_list_after)
    imageio.mimsave(os.path.join(out_dir, 'succ_gif', '%d_%s_%s_%s.gif' % (trial_id, scene, category, shape_id)), gif_imgs)

def save_fail_data(out_info, cam_XYZA_list, gt_link_mask, cam_XYZA_list_after, gt_link_mask_after, gif_imgs):
    if args.save_data:
        utils.save_data(os.path.join(out_dir, 'fail_files'), trial_id, out_info, cam_XYZA_list, gt_link_mask, gt_link_mask_after=gt_link_mask_after, final_cam_XYZA_list=cam_XYZA_list_after)
        imageio.mimsave(os.path.join(out_dir, 'fail_gif', '%d_%s_%s_%s.gif' % (trial_id, scene, category, shape_id)), gif_imgs)

def save_invalid_data(out_info, cam_XYZA_list, gt_link_mask, fimg):
    # if args.save_data:
    utils.save_data(os.path.join(out_dir, 'invalid_files'), trial_id, out_info, cam_XYZA_list, gt_link_mask)
    fimg.save(os.path.join(out_dir, 'invalid_gif', '%d_%s_%s_%s.png' % (trial_id, scene, category, shape_id)))

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
    thinkness = 0.1 + np.random.uniform(-0.05, 0.05)
    height = 1.0
    # have greater probablity for larger values
    x_zone = 1 if np.random.randint(2) == 0 else -1
    y_zone = 1 if np.random.randint(2) == 0 else -1
    x_offset = np.random.uniform()**1.5*size_x*0.45*x_zone
    y_offset = np.random.uniform()**1.5*size_y*0.45*y_zone
    z_offset = 0.06 + height

    env = Env(show_gui=(not args.no_gui), set_ground=True, object_position_offset=[x_offset, y_offset, z_offset])
    env.create_table(Pose([0, 0, 0]), size_x, size_y, height, thinkness, (0.8, 0.6, 0.4), 1.0, 0.01, 1000, 'table')
    dist = 3.6

    # dist=5
    cam = Camera(env, fixed_position=False, dist=dist, have_table=have_table, table_height=height, pos_rand_ratio=0., object_centered=False)
    print("camera created")
    check_strict = True

    obj_random_orientation = 0.2


if args.scene == 'wall':
    have_table = False
    
    x_offset = np.random.uniform()*0.8 + 0.3
    z_offset = 0.06
    gravity = [0., 0, -9.81]
    y_offset = 0.

    env = Env(show_gui=(not args.no_gui), set_ground=True, object_position_offset=[x_offset, y_offset, z_offset], gravity=gravity)
    env.create_box(Pose([-0.5, 0, 0.5], [1,0,0,0]), [0.5,2.0,0.5], (0.8, 0.6, 0.4), friction=0.3, restitution=0.01, density=1000, name='wall')
    dist = 3.6
    check_strict = True

    cam = Camera(env, fixed_position=False, dist=dist, pos_rand_ratio=0., object_centered=False)
    print("camera created")
    obj_random_orientation = 0.2


if args.scene == 'groove':
    have_table = False
    
    x_offset = 0.3+np.random.uniform()*0.5
    z_offset = 0.16

    y_offset = 0.
    env = Env(show_gui=(not args.no_gui), set_ground=True, object_position_offset=[x_offset, y_offset, z_offset])
    groove_urdf = r'../../Shape_data/env_assets/groove_wide/urdf/groove_wide.urdf'
    env.create_groove()
    dist = 3.6
    check_strict = True
    cam = Camera(env, fixed_position=False, dist=dist, pos_rand_ratio=0., object_centered=False)
    print("camera created")

    obj_random_orientation = 0.2

    
if args.scene == 'slope':
    have_table = False
    
    x_offset = np.random.uniform()*0.5 + 0.3
    z_offset = 0.06

    y_offset = 0.
    
    env = Env(show_gui=(not args.no_gui), set_ground=True, object_position_offset=[x_offset, y_offset, z_offset])
    groove_urdf = r'../../Shape_data/env_assets/slope/urdf/slope.urdf'
    env.create_slope(groove_urdf, Pose([0,0,0], [0.5**0.5, -0.5**0.5, 0, 0]), scale=3.0)
    # dist = np.random.uniform(4, 4.5)
    dist = 3.6
    check_strict = True

    # dist=5
    cam = Camera(env, fixed_position=False, dist=dist, pos_rand_ratio=0., object_centered=False)
    print("camera created")
    obj_random_orientation = 0.2


if args.scene == 'multiple':
    have_table = True
    
    x_offset = np.random.uniform(-0.6, 0.6)
    z_offset = 1.1
    y_offset = np.random.uniform(-0.6, 0.6)
    
    env = Env(show_gui=(not args.no_gui), set_ground=True, object_position_offset=[x_offset, y_offset, z_offset])
    env.create_multiple_scenes()
    dist = np.random.uniform(3., 3.5)
    check_strict = False
    dist=3.6
    cam = Camera(env, fixed_position=False, dist=dist, pos_rand_ratio=0., object_centered=False, have_table=have_table)
    print("camera created")
    obj_random_orientation = 0.2


if not args.no_gui:
    env.set_controller_camera_pose(cam.pos[0], cam.pos[1], cam.pos[2], np.pi + cam.theta, -cam.phi)

specific_categories = ['Phone', 'Tablet']
size_dict = {'22001': 0.00035, '22002': 0.0015, '22003': 0.0005, '22004': 0.013, '30001': 1.0, '30002': 0.002, '30003': 0.2, '30004': 0.0015, '30005': 0.002}
mini_categories = ['Switch', 'Pen', 'USB', 'Bowl', 'Remote', 'Bowl_lie']
smaller_categories = ['Box', 'Display', 'Bucket', 'Eyeglasses', 'Faucet', 'Kettle', 'Keyboard', 'KitchenPot', 'Laptop', 'Microwave', 'Oven', 'Pliers', 'Scissors', 'Toaster', 'Window', 'Basket', 'Keyboard2', 'Cap']
medium_categories = ['Dishwasher', 'Printer', 'Refrigerator', 'Safe', 'StorageFurniture', 'Basket', 'Chair2']
Large_categories = ['Cart', 'Chair', 'Door', 'FoldingChair', 'Table', 'TrashCan', 'WashingMachine', 'Bench', 'Sofa']

PartNet_categories = ['Phone', 'Box', 'Bucket', 'Cart', 'Chair', 'Dishwasher', 'Display', 'Door', 'Eyeglasses', 'Faucet', 'FoldingChair', 'Kettle', 'Keyboard', 'KitchenPot', 'Laptop', 'Microwave', 'Oven', 'Pen', 'Pliers', 'Printer', 'Refrigerator', 'Remote', 'Safe', 'Scissors', 'StorageFurniture', 'Switch', 'Table', 'Toaster', 'TrashCan', 'USB', 'WashingMachine', 'Window','Tablet']
ShapeNet_categories = ['Basket', 'Bench', 'Bowl', 'Bowl_lie', 'Chair2', 'Keyboard2', 'Sofa', 'Cap']

Thin_categories = ['Switch', 'Laptop', 'Remote', 'Scissors', 'Window', 'Keyboard2', 'Pen', 'USB', 'Bowl_lie', 'Cap', 'Phone','Tablet']
Pickable_categories = ['Box', 'Bucket', 'Display', 'Eyeglasses', 'Faucet', 'Kettle', 'KitchenPot', 'Pliers', 'Basket', 'Bowl']
Unpickable_categories = ['Cart', 'Chair', 'Dishwasher', 'Door', 'FoldingChair', 'Microwave', 'Oven', 'Printer', 'Refrigerator', 'Safe', 'StorageFurniture', 'Table', 'Toaster', 'TrashCan', 'WashingMachine', 'Bench', 'Chair2', 'Sofa']
Liedown_categories = ['Switch', 'Remote', 'Window', 'Pen', 'Phone']
Down_categories = ['Bowl_lie']

Middle_state_categories = ['Display']

cat_list, shape_list, shape2cat_dict, cat2shape_dict = utils.get_shape_list(all_categories=args.category, mode='all')
if shape_id is None:
    shape_id = cat2shape_dict[category][np.random.random_integers(0, high=len(cat2shape_dict[category])-1)]
    print(cat2shape_dict[category])

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
        joint_angles = env.load_object(object_urdf_fn, object_material, state=args.target_part_state, scale=0.15, density=args.density, damping=args.damping, stiffness=5., rand_orientation=obj_random_orientation, lieDown=Liedown, down=Down)
    elif category in smaller_categories:
        joint_angles = env.load_object(object_urdf_fn, object_material, state=args.target_part_state, scale=0.28, density=args.density, damping=args.damping, stiffness=5., rand_orientation=obj_random_orientation, lieDown=Liedown, down=Down)
    elif category in medium_categories:
        joint_angles = env.load_object(object_urdf_fn, object_material, state=args.target_part_state, scale=0.75, density=args.density, damping=args.damping, stiffness=5., rand_orientation=obj_random_orientation, lieDown=Liedown, down=Down)
    elif category in specific_categories:
        joint_angles = env.load_object(object_urdf_fn, object_material, state=args.target_part_state, scale=size_dict[args.shape_id], density=args.density, damping=args.damping, stiffness=5., rand_orientation=obj_random_orientation, lieDown=Liedown, down=Down)
    else:
        joint_angles = env.load_object(object_urdf_fn, object_material, state=args.target_part_state, scale=1.0, density=args.density, damping=args.damping, stiffness=5., rand_orientation=obj_random_orientation, lieDown=Liedown, down=Down)

except Exception:
    print('error while load object')
    for a in env.scene.get_all_actors():
        env.scene.remove_actor(a)
    env.close()
    exit(4)

# wait for the object's still
# qp_monitor = monitor.MonitorApp(window_title='qp', interval=10)
qp_monitor = None
still_timesteps = utils.wait_for_object_still(env, have_table=have_table, monitor=qp_monitor)


if still_timesteps < 200:
    print('Object Not Still!')
    env.scene.remove_articulation(env.object)
    for a in env.scene.get_all_actors():
        env.scene.remove_actor(a)
    env.close()
    # write category, shape_id
    with open('./wrong_obj.txt', 'a') as fout:
        fout.write('%s %s\n' % (category, shape_id))
    exit(5)


### use the GT vision
rgb, depth = cam.get_observation()
cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = cam.compute_camera_XYZA(depth) # return the point cloud between near and far
object_movable_link_ids = env.movable_link_ids
object_all_link_ids = env.all_link_ids
print('object_movable_link_ids = ', object_movable_link_ids)
cam_XYZA = cam.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, depth.shape[0], depth.shape[1], object_all_link_ids) # [depth.shape[0], depth.shape[1], 4]

cam_XYZA_list = [cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, cam_XYZA]

# pc, pc_centers = utils.get_part_pc(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, 'world', mat44=np.array(cam.get_metadata_json()['mat44'], dtype=np.float32))
# pc = pc.detach().cpu().numpy().reshape(-1, 3)
print('cam_XYZA', cam_XYZA)
if scene != 'multiple':
    con = utils.check_con(env, cam, cam_XYZA, scene)
else:
    con, con_type = utils.check_con(env, cam, cam_XYZA, scene)
    print('con_type = ', con_type)
if con:
    print('con at beginning')
    env.scene.remove_articulation(env.object)
    for a in env.scene.get_all_actors():
        env.scene.remove_actor(a)
    env.close()
    exit(109)

gt_movable_link_mask = cam.get_movable_link_mask(object_movable_link_ids)  # (448, 448), 0(unmovable) - id(movable)
gt_all_link_mask = cam.get_movable_link_mask(object_all_link_ids)  # (448, 448), 0(unmovable) - id(all)

# sample a pixel on target part
xs, ys = np.where(gt_all_link_mask > 0)
if len(xs) == 0:
    env.scene.remove_articulation(env.object)
    env.close()
    print("no_target_point")
    exit(6)

if_edge = (np.random.uniform() < 0.75 and args.category in Thin_categories and args.use_edge)
if if_edge:
    x, y, top_10 = utils.find_edge_point(xs, ys)
else:
    idx = np.random.randint(len(xs))
    x, y = xs[idx], ys[idx]

target_part_id = object_all_link_ids[gt_all_link_mask[x, y] - 1]
env.set_target_object_part_actor_id2(target_part_id)  # for get_target_part_pose

# for j in env.object.get_joints():
#     print(j.type, j.get_name(), j.damping, j.friction, j.get_child_link(), j.get_dof(), j.get_parent_link())
# to find a link with fixed joint
target_joint_type = ArticulationJointType.FIX
tot_trial = 0
while tot_trial < 50 and (env.target_object_part_joint_type != target_joint_type):
    if if_edge:
        chosen_index = np.random.choice(top_10)
        x, y = xs[chosen_index], ys[chosen_index]
    else:
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
# obj_pose_2 = env.get_object_root_pose()
obj_root_pose = env.object.get_pose()
print('obj_start_pose = ', obj_root_pose)

root_x = obj_root_pose.p[0]
root_y = obj_root_pose.p[1]

if scene == 'table' or 'multiple':
    x_zone = 1 if root_x > 0 else -1
    y_zone = 1 if root_y > 0 else -1
    if args.if_given_direction:
        if abs(0.75*x_zone - root_x) > abs(0.75*y_zone - root_y):
            task = np.array([0, 0.75*y_zone - root_y])*0.85 + np.random.uniform(-0.05, 0.05, size=2)
        else:
            task = np.array([0.75*x_zone - root_x, 0])*0.85 + np.random.uniform(-0.05, 0.05, size=2)
    print('table task = ', task)

if scene == 'wall':
    if args.if_given_direction:
        if category in mini_categories:
            offset = 0.04
        elif category in smaller_categories:
            offset = 0.1
        elif category in specific_categories:
            offset = 0.1
        else:
            offset = 0.2
        task = np.array([-root_x+offset, 0]) + np.random.uniform(-0.3, 0.1, size=2)
        print('task = ', task)

if scene == 'groove':
    if args.if_given_direction:
        task = np.array([-root_x+0.05, 0]) + np.random.uniform(-0.2, 0.1, size=2)
        print('task = ', task)

if scene == 'slope':
    if args.if_given_direction:
        task = np.array([-root_x+0.05, 0]) + np.random.uniform(-0.05, 0.05, size=2)

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

still_joint_angles = env.object.get_qpos()

contact_trail = 0
env.contact_error = None
if_pose_changed = False
while True:
    env.render()
    idx1 = np.random.randint(len(xs))
    x1, y1 = xs[idx1], ys[idx1]

    env.object.set_pose(obj_root_pose)
    env.object.set_root_velocity(np.zeros(6))
    env.object.set_qpos(still_joint_angles)
    env.object.set_qvel(np.zeros_like(still_joint_angles))
    # print(env.object.get_drive_target(), env.object.get_qpos())
    still_timesteps = utils.wait_for_object_still(env, have_table=have_table)

    after_still_joint_angles = env.object.get_qpos()
    after_still_root_pose = env.object.get_pose()
    # with open('./displacement.txt', 'a') as fout:
    #     fout.write('%f %f\n' % (after_still_joint_angles, after_still_root_pose.p))
    if len(after_still_joint_angles) > 0:
        max_q_displacement = np.max(np.abs(after_still_joint_angles - still_joint_angles))
    else:
        max_q_displacement = 0
    max_root_displacement = np.max(np.abs(after_still_root_pose.p - obj_root_pose.p))

    if max_q_displacement > 1e-2:
        print("q still error")
        print("max_q_displacement = ", max_q_displacement)
        # env.scene.remove_articulation(env.object)
        # env.close()
        # exit(8)
        if_pose_changed = True
    if max_root_displacement > 1e-2:
        print("root still error")
        print("max_root_displacement = ", max_root_displacement)
        # env.scene.remove_articulation(env.object)
        # env.close()
        # exit(9)
        if_pose_changed = True

    if if_pose_changed:
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

        if_edge = (np.random.uniform() < 0.75 and args.category in Thin_categories)
        if if_edge:
            x, y, top_10 = utils.find_edge_point(xs, ys)
        else:
            idx = np.random.randint(len(xs))
            x, y = xs[idx], ys[idx]

        target_part_id = object_all_link_ids[gt_all_link_mask[x, y] - 1]
        env.set_target_object_part_actor_id2(target_part_id)  # for get_target_part_pose

        # for j in env.object.get_joints():
        #     print(j.type, j.get_name(), j.damping, j.friction, j.get_child_link(), j.get_dof(), j.get_parent_link())
        # to find a link with fixed joint
        target_joint_type = ArticulationJointType.FIX
        tot_trial = 0
        while tot_trial < 50 and (env.target_object_part_joint_type != target_joint_type):
            if if_edge:
                chosen_index = np.random.choice(top_10)
                x, y = xs[chosen_index], ys[chosen_index]
            else:
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
        # obj_pose_2 = env.get_object_root_pose()
        obj_root_pose = env.object.get_pose()
        # print("obj_pose = ", obj_pose)
        # print("obj_pose_2 = ", obj_pose_2)
        # print("obj_root_pose = ", obj_root_pose)

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
        still_joint_angles = env.object.get_qpos()
        if_pose_changed = False
        continue

    # move back
    # env.render()
    # if not args.no_gui:
    #     env.wait_to_start()
    #     pass

    if not args.if_given_direction:
        theta = np.random.uniform(0, 2*np.pi)
        task = (abs(np.random.normal(0, 0.2))+0.05) * np.array([np.cos(theta), np.sin(theta)])
    given_left = np.array([0,0,-1])
    if_given_left = True
    if_given_up = True
    direct = task / np.linalg.norm(task)
    direct = np.array([direct[0], direct[1], 0])
    given_up = np.array([direct[0]*3**0.5/2, direct[1]*3**0.5/2, -0.5])
    pre_pose, pre_rotmat, start_pose, start_rotmat, final_pose, final_rotmat, up, forward = utils.cal_final_pose(cam, cam_XYZA, x1, y1, number='1', out_info=out_info, start_dist=args.start_dist, maneuver_dist=args.maneuver_dist, displacement=np.linalg.norm(task), if_given_forward=False, given_forward=np.array([0,0,1]), if_given_left = if_given_left, given_left = given_left, action_direction_world=direct, if_constraint_up = False, given_up = given_up, if_given_up=if_given_up)
    # print("up = ", up, "forward = ", forward)
    # print("pre_pose = ", pre_pose, "pre_rotmat = ", pre_rotmat)
    print("pose = ", start_pose, final_pose)

    # setup robot
    robot_urdf_fn = './airbot/urdf_obj/AIRBOT_hand.urdf'
    robot_material = env.get_material(1., 1., 0.01)
    robot_scale = 1.2

    robot = Robot(env, robot_urdf_fn, robot_material, open_gripper=True, scale=robot_scale, if_fixed=True, if_push = True)
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
    # fimg.save(f"./fimg{contact_trail}.png")
    # gif_imgs.extend(fimg)
    # print the caught exception
    env.step()


    if not env.contact_error:
        print("valid contact")
        break

    if contact_trail > 2:
        print("contact error not solved")
        out_info['random_seed'] = args.random_seed
        out_info['pixel_locs'] = [int(x), int(y)]
        out_info['target_object_part_joint_type'] = str(env.target_object_part_joint_type)
        out_info['camera_metadata'] = cam.get_metadata_json()
        out_info['object_state'] = args.target_part_state
        out_info['joint_angles'] = joint_angles
        out_info['still_joint_angles'] = still_joint_angles.tolist()
        out_info['joint_angles_lower'] = env.joint_angles_lower
        out_info['joint_angles_upper'] = env.joint_angles_upper
        out_info['shape_id'] = shape_id
        out_info['category'] = category
        out_info['scene'] = scene
        out_info['pixel1_idx'] = int(idx1)
        out_info['target_link_mat44'] = pre_target_link_mat44.tolist()
        out_info['prev_origin_world'] = prev_origin_world.tolist()
        out_info['obj_pose_p'] = obj_pose.p.tolist()
        out_info['obj_pose_q'] = obj_pose.q.tolist()
        out_info['task'] = task.tolist()
        out_info['success'] = 'False'
        # out_info['success_bd'] = 'False'
        out_info['result'] = 'INVALID'
        save_invalid_data(out_info, cam_XYZA_list, gt_all_link_mask, fimg)
        env.scene.remove_articulation(env.object)
        env.scene.remove_articulation(robot.robot)
        for a in env.scene.get_all_actors():
            env.scene.remove_actor(a)
        env.close()
        exit(3)

    print(f'contact error {contact_trail}')
    env.scene.remove_articulation(robot.robot)
    contact_trail+=1

out_info['random_seed'] = args.random_seed
out_info['pixel_locs'] = [int(x), int(y)]
out_info['target_object_part_joint_type'] = str(env.target_object_part_joint_type)
out_info['camera_metadata'] = cam.get_metadata_json()
out_info['object_state'] = args.target_part_state
out_info['joint_angles'] = joint_angles
out_info['still_joint_angles'] = still_joint_angles.tolist()
out_info['joint_angles_lower'] = env.joint_angles_lower
out_info['joint_angles_upper'] = env.joint_angles_upper
out_info['shape_id'] = shape_id
out_info['category'] = category
out_info['scene'] = scene
out_info['task'] = task.tolist()
out_info['pixel1_idx'] = int(idx1)
out_info['target_link_mat44'] = pre_target_link_mat44.tolist()
out_info['prev_origin_world'] = prev_origin_world.tolist()
out_info['obj_pose_p'] = obj_pose.p.tolist()
out_info['obj_pose_q'] = obj_pose.q.tolist()
out_info['success'] = 'False'
# out_info['success_bd'] = 'False'
out_info['result'] = 'VALID'

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
env.check_contact_strict = True

try:
    print("start maneuver")
    imgs = utils.gripper_move_to_target_pose(robot, start_rotmat, num_steps=args.maneuver_steps, cam=cam, vis_gif=True, vis_gif_interval=100, monitor = qf_monitor)
    gif_imgs.extend(imgs)
    print("start waiting")
    imgs = utils.gripper_wait_n_steps(robot, n=args.wait_steps, cam=cam, vis_gif=True, vis_gif_interval=100, monitor = qf_monitor)
    gif_imgs.extend(imgs)
    # print(len(imgs))
    print("start pushing")
    robot_start_pose = robot.end_effector.get_pose().p
    print("robot start position", robot.end_effector.get_pose())
    imgs = utils.gripper_move_to_target_pose(robot, final_rotmat, num_steps=args.move_steps, cam=cam, vis_gif=True, vis_gif_interval=100, monitor = qf_monitor)
    gif_imgs.extend(imgs)
    # print(len(imgs))
    print("start waiting")
    imgs = utils.gripper_wait_n_steps(robot, n=args.wait_steps, cam=cam, vis_gif=True, vis_gif_interval=100, monitor = qf_monitor)
    gif_imgs.extend(imgs)
    env.end_checking_contact(robot.hand_actor_id, robot.gripper_actor_ids, False)
    print("end checking contact")
    print(len(imgs))
    robot_end_pose = robot.end_effector.get_pose().p
    print("robot final position", robot.end_effector.get_pose())
    env.scene.remove_articulation(robot.robot)
    still_timesteps = utils.wait_for_object_still(env, have_table=have_table)
    if still_timesteps < 200:
        print('Object Not Still!')
        env.scene.remove_articulation(env.object)
        for a in env.scene.get_all_actors():
            env.scene.remove_actor(a)
        env.close()
        exit(5)

except Exception:
    print('contact error')
    out_info['result'] = 'INVALID'
    save_invalid_data(out_info, cam_XYZA_list, gt_all_link_mask, fimg)
    env.scene.remove_articulation(env.object)
    env.scene.remove_articulation(robot.robot)
    for a in env.scene.get_all_actors():
        env.scene.remove_actor(a)
    env.close()
    exit(3)

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

next_obj_root_pose = env.object.get_pose()
print('obj_next_pose = ', next_obj_root_pose)
out_info['next_obj_root_pose_p'] = next_obj_root_pose.p.tolist()
out_info['next_obj_root_pose_q'] = next_obj_root_pose.q.tolist()
out_info['next_still_joint_angles'] = env.object.get_qpos().tolist()
# calculate displacement
next_origin_world_xyz1 = target_part_trans @ np.array([0, 0, 0, 1])
next_origin_world = next_origin_world_xyz1[:3]
trajectory = next_origin_world - prev_origin_world
# print('before check success')
traj_robot = robot_end_pose - robot_start_pose
# print('after check success')
rgb, depth = cam.get_observation()
after_cam_XYZA_id1, after_cam_XYZA_id2, after_cam_XYZA_pts = cam.compute_camera_XYZA(depth) # return the point cloud between near and far
after_cam_XYZA = cam.compute_XYZA_matrix(after_cam_XYZA_id1, after_cam_XYZA_id2, after_cam_XYZA_pts, depth.shape[0], depth.shape[1], object_all_link_ids) # [depth.shape[0], depth.shape[1], 4]
cam_XYZA_list_after = [after_cam_XYZA_id1, after_cam_XYZA_id2, after_cam_XYZA_pts, after_cam_XYZA]
gt_all_link_mask_after = cam.get_movable_link_mask(object_all_link_ids)  # (448, 448), 0(unmovable) - id(all)

if scene != 'multiple':
    con = utils.check_con(env, cam, after_cam_XYZA, scene)
else:
    con, con_type = utils.check_con(env, cam, after_cam_XYZA, scene)

success, traj_x, traj_y, traj_z = utils.check_success(trajectory, alpha, beta, gamma, task=task, threshold=args.threshold, threshold_t=args.threshold_t, if_push=True, traj_robot = traj_robot, con=con, scene=scene)

out_info['success'] = 'True' if (success and args.if_given_direction) else 'False'
out_info['traj_robot'] = traj_robot.tolist()
out_info['trajectory'] = trajectory.tolist()
out_info['if_given_direction'] = args.if_given_direction
out_info['traj_x'] = traj_x.tolist()
out_info['traj_y'] = traj_y.tolist()
out_info['traj_z'] = traj_z.tolist()
out_info['con'] = con

env.scene.remove_articulation(env.object)
for a in env.scene.get_all_actors():
        env.scene.remove_actor(a)
env.close()

if success:
    print('success')
    save_succ_data(out_info, cam_XYZA_list, gt_all_link_mask, cam_XYZA_list_after, gt_all_link_mask_after, gif_imgs)
    exit(0)
else:
    print('failed')
    save_fail_data(out_info, cam_XYZA_list, gt_all_link_mask, cam_XYZA_list_after, gt_all_link_mask_after, gif_imgs)
    if args.if_given_direction:
        exit(1)
    else:
        exit(2)
