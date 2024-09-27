import os
import sys
import h5py
import torch
import numpy as np
import importlib
import random
import shutil
import math
from PIL import Image
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
from subprocess import call
from sapien.core import Pose
import json
import torch.nn.functional as F
from pointnet2_ops.pointnet2_utils import furthest_point_sample
from sapien.core import Pose, ArticulationJointType
from camera import Camera
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import subprocess


class ContactError(Exception):
    pass


class DivisionError(Exception):
    pass


def printout(flog, strout):
    print(strout)
    if flog is not None:
        flog.write(strout + '\n')


def optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def get_model_module(model_version):
    importlib.invalidate_caches()
    return importlib.import_module('models.' + model_version)


def collate_feats(b):
    return list(zip(*b))


def worker_init_fn(worker_id):
    """ The function is designed for pytorch multi-process dataloader.
        Note that we use the pytorch random generator to generate a base_seed.
        Please try to be consistent.
        References:
            https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed
    """
    base_seed = torch.IntTensor(1).random_().item()
    # print(worker_id, base_seed)
    np.random.seed(base_seed + worker_id)


def export_pts(out, v):
    with open(out, 'w') as fout:
        for i in range(v.shape[0]):
            fout.write('%f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2]))


def export_label(out, l):
    with open(out, 'w') as fout:
        for i in range(l.shape[0]):
            fout.write('%f\n' % (l[i]))


def export_pts_label(out, v, l):
    with open(out, 'w') as fout:
        for i in range(l.shape[0]):
            fout.write('%f %f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2], l[i]))


def render_pts_label_png_modified(out, v, l, mat44, cam2cambase, pc_center):
    is_pc = np.ones(v.shape[0])
    cam_batch = batch_coordinate_transform(v, is_pc, 'cambase2cam', mat44, cam2cambase, pc_center)
    world_batch = batch_coordinate_transform(cam_batch, is_pc, 'cam2world', mat44, cam2cambase, pc_center)
    v_world = np.array(world_batch)
    render_pts_label_png(out, v_world, l)
    # Create a 3D scatter plot
    # use mitsuba ...
    return ...

def render_pts_label_png(out, v, l):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(v[:, 0], v[:, 1], v[:, 2], c=l, s=1)

    X = v[:, 0]
    Y = v[:, 1]
    Z = v[:, 2]
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # Save the plot to a file
    plt.savefig(f'{out}.png', dpi=300)
    plt.close(fig)


def render_proposal_png(out, v, l, ctpt, up, forward=None): # v:points, l:color, up, forward 
    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(v[:, 0], v[:, 1], v[:, 2], c=l, s=1)

    if forward is None:
        length = np.linalg.norm(up)
    else:
        length = 0.5
    ax.quiver(ctpt[0], ctpt[1], ctpt[2], up[0], up[1], up[2], 
            color='green', length=length, normalize=True)
    if forward is not None:
        ax.quiver(ctpt[0], ctpt[1], ctpt[2], forward[0], forward[1], forward[2], 
            color='blue', length=0.5, normalize=True)

    X = v[:, 0]
    Y = v[:, 1]
    Z = v[:, 2]
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # Save the plot to a file
    plt.savefig(f'{out}.png', dpi=300)
    plt.close(fig)


def render_png_given_pts(out):
    cmd = 'xvfb-run -a ~/thea/TheaDepsUnix/Source/TheaPrefix/bin/Thea/RenderShape %s.pts -f %s.feats %s.png 448 448 -v 1,0,0,-5,0,0,0,0,1 >> /dev/null' % (out, out, out)
    call(cmd, shell=True)


def export_pts_color_obj(out, v, c):
    with open(out + '.obj', 'w') as fout:
        for i in range(v.shape[0]):
            fout.write('v %f %f %f %f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2], c[i, 0], c[i, 1], c[i, 2]))


def export_pts_color_pts(out, v, c):
    with open(out + '.pts', 'w') as fout:
        for i in range(v.shape[0]):
            fout.write('%f %f %f %f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2], c[i, 0], c[i, 1], c[i, 2]))


def rot2so3(rotation):
    assert rotation.shape == (3, 3)
    if np.isclose(rotation.trace(), 3):
        return np.zeros(3), 1
    if np.isclose(rotation.trace(), -1):
        raise RuntimeError
    theta = np.arccos((rotation.trace() - 1) / 2)
    omega = 1 / 2 / np.sin(theta) * np.array(
        [rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]]).T
    return omega, theta


def skew(vec):
    return np.array([[0, -vec[2], vec[1]],
                     [vec[2], 0, -vec[0]],
                     [-vec[1], vec[0], 0]])


def adjoint_matrix(pose):
    adjoint = np.zeros([6, 6])
    adjoint[:3, :3] = pose[:3, :3]
    adjoint[3:6, 3:6] = pose[:3, :3]
    adjoint[3:6, 0:3] = skew(pose[:3, 3]) @ pose[:3, :3]
    return adjoint


def pose2exp_coordinate(pose):
    """
    Compute the exponential coordinate corresponding to the given SE(3) matrix
    Note: unit twist is not a unit vector

    Args:
        pose: (4, 4) transformation matrix

    Returns:
        Unit twist: (6, ) vector represent the unit twist
        Theta: scalar represent the quantity of exponential coordinate
    """

    omega, theta = rot2so3(pose[:3, :3])
    ss = skew(omega)
    inv_left_jacobian = np.eye(3, dtype=np.float) / theta - 0.5 * ss + (
            1.0 / theta - 0.5 / np.tan(theta / 2)) * ss @ ss
    v = inv_left_jacobian @ pose[:3, 3]
    return np.concatenate([omega, v]), theta


def process_angle_limit(x, type):
    if np.isneginf(x):
        x = -10
        if type == ArticulationJointType.REVOLUTE:
            x = -2*np.pi
            
    if np.isinf(x):
        x = 10
        if type == ArticulationJointType.REVOLUTE:
            x = 2*np.pi
    return x


def get_random_number(l, r):
    return np.random.rand() * (r - l) + l


def save_h5(fn, data):
    fout = h5py.File(fn, 'w')
    for d, n, t in data:
        fout.create_dataset(n, data=d, compression='gzip', compression_opts=4, dtype=t)
    fout.close()


def calc_part_motion_degree(part_motion):
    return part_motion * 180.0 / 3.1415926535


def radian2degree(radian):
    return radian * 180.0 / np.pi


def degree2radian(degree):
    return degree / 180.0 * np.pi


def cal_Fscore(pred, labels):
    TP, TN, FN, FP = 0, 0, 0, 0
    TP += ((pred == 1) & (labels == 1)).sum()  
    TN += ((pred == 0) & (labels == 0)).sum()  
    FN += ((pred == 0) & (labels == 1)).sum()  
    FP += ((pred == 1) & (labels == 0)).sum()  
    try:
        p = TP / (TP + FP)
    except:
        p = 0
    try:
        r = TP / (TP + FN)
    except:
        r = 0
    try:
        F1 = 2 * r * p / (r + p)
    except:
        F1 = 0

    acc = (pred == labels).sum() / len(pred)
    return F1, p, r, acc


def cal_included_angle(x, y):
    len_x = np.linalg.norm(x)
    len_y = np.linalg.norm(y)
    cos_ = (x @ y) / (len_x * len_y)
    angle_radian = np.arccos(np.clip(cos_, -1 + 1e-6, 1 - 1e-6))
    angle_degree = angle_radian * 180 / np.pi
    len_projection = len_x * cos_   # the projection of x on y
    return angle_degree, len_projection


def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    sy = math.sqrt(R[2, 1] * R[2, 1] + R[2, 2] * R[2, 2])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])    # pitch (Y-axis)
        y = math.atan2(-R[2, 0], sy)        # yaw (Z-axis)
        z = math.atan2(R[1, 0], R[0, 0])    # roll (X-axis)
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x * 180 / np.pi, y * 180 / np.pi, z * 180 / np.pi])


def get_contact_point(cam, cam_XYZA, x, y):
    position_cam = cam_XYZA[x, y, :3]
    position_cam_xyz1 = np.ones((4), dtype=np.float32)
    position_cam_xyz1[:3] = position_cam
    position_world_xyz1 = cam.get_metadata()['mat44'] @ position_cam_xyz1
    position_world = position_world_xyz1[:3]
    return position_world, position_cam, position_world_xyz1


def get_rotmat_full(cam, position_world, up, forward, number, out_info, start_dist=0.20, final_dist=0.08, act_type=0, term=""):
    # run bgs before runing get_rotmat
    left = np.cross(up, forward)
    left /= np.linalg.norm(left)
    forward = np.cross(left, up)
    forward /= np.linalg.norm(forward)
    forward_cam = np.linalg.inv(cam.get_metadata()['mat44'][:3, :3]) @ forward
    up_cam = np.linalg.inv(cam.get_metadata()['mat44'][:3, :3]) @ up
    out_info['position_world' + term + number] = position_world.tolist()  # world
    out_info['gripper_direction_world' + term + number] = up.tolist()
    out_info['gripper_direction_camera' + term + number] = up_cam.tolist()
    out_info['gripper_forward_direction_world' + term + number] = forward.tolist()
    out_info['gripper_forward_direction_camera' + term + number] = forward_cam.tolist()
    rotmat = np.eye(4).astype(np.float32)  # rotmat: world coordinate
    rotmat[:3, 0] = forward
    rotmat[:3, 1] = left
    rotmat[:3, 2] = up

    pos2_rotmat = np.array(rotmat, dtype=np.float32)
    pos2_rotmat[:3, 3] = position_world - up * final_dist
    pos2_pose = Pose().from_transformation_matrix(pos2_rotmat)
    out_info['target_rotmat_world' + term + number] = pos2_rotmat.tolist()

    pos1_rotmat = np.array(rotmat, dtype=np.float32)
    pos1_rotmat[:3, 3] = position_world - up * start_dist
    pos1_pose = Pose().from_transformation_matrix(pos1_rotmat)
    out_info['start_rotmat_world' + term + number] = pos1_rotmat.tolist()

    pos3_rotmat = np.array(rotmat, dtype=np.float32)
    pos3_rotmat[:3, 3] = position_world - up * start_dist
    if act_type == 1:
        pos3_rotmat[2, 3] = pos2_rotmat[2, 3]

    return pos1_pose, pos1_rotmat, pos2_pose, pos2_rotmat, pos3_rotmat


def get_rotmat(cam, position_world, up, forward=None, number=None, out_info=None, start_dist=0.20, final_dist=0.08, if_given_left=False, given_left=None):
    if not if_given_left:
        up /= np.linalg.norm(up)
        left = np.cross(up, forward)
        left /= np.linalg.norm(left)
        forward = np.cross(left, up)
        forward /= np.linalg.norm(forward)
    else:
        up /= np.linalg.norm(up)
        left = given_left
        while (up @ left) > 0.99:
            left = np.random.randn(3).astype(np.float32)
        forward = np.cross(left, up)
        forward /= np.linalg.norm(forward)
        left = np.cross(up, forward)
        left /= np.linalg.norm(left)
    forward_cam = np.linalg.inv(cam.get_metadata()['mat44'][:3, :3]) @ forward
    up_cam = np.linalg.inv(cam.get_metadata()['mat44'][:3, :3]) @ up
    
    rotmat = np.eye(4).astype(np.float32)  # rotmat: world coordinate
    rotmat[:3, 0] = forward
    rotmat[:3, 1] = left
    rotmat[:3, 2] = up
    # print(rotmat)

    final_rotmat = np.array(rotmat, dtype=np.float32)
    final_rotmat[:3, 3] = position_world - up * final_dist
    final_pose = Pose().from_transformation_matrix(final_rotmat)
    

    start_rotmat = np.array(rotmat, dtype=np.float32)
    
    start_rotmat[:3, 3] = position_world - up * start_dist
    # print(start_rotmat)
    start_pose = Pose().from_transformation_matrix(start_rotmat)
    
    if number is not None:
        out_info['position_world' + number] = position_world.tolist()  # world
        out_info['gripper_direction_world' + number] = up.tolist()
        out_info['gripper_direction_camera' + number] = up_cam.tolist()
        out_info['gripper_forward_direction_world' + number] = forward.tolist()
        out_info['gripper_forward_direction_camera' + number] = forward_cam.tolist()
        out_info['target_rotmat_world' + number] = final_rotmat.tolist()
        out_info['start_rotmat_world' + number] = start_rotmat.tolist()

    return start_pose, start_rotmat, final_pose, final_rotmat


def cal_final_pose(cam, cam_XYZA, x, y, number, out_info, start_dist=0.12, maneuver_dist=0.18, start_pose=None, start_rotmat=None, displacement=0.4, if_given_forward=False, given_forward = None, if_given_up=False, given_up=None, if_given_left = False, given_left = None, action_direction_world = np.array([0,0,1]), if_constraint_up = False):
    # given_up: action_direction
    # given_forward: rotation around action_direction
    # get pixel 3D position (cam/world)
    position_cam = cam_XYZA[x, y, :3]   # contact point
    position_cam_xyz1 = np.ones((4), dtype=np.float32)
    position_cam_xyz1[:3] = position_cam
    position_world_xyz1 = cam.get_metadata()['mat44'] @ position_cam_xyz1
    position_world = position_world_xyz1[:3]
    out_info['position_cam' + number] = position_cam.tolist()   # contact point at camera c
    out_info['position_world' + number] = position_world.tolist()   # world

    # get pixel 3D pulling direction (normal vector) (cam/world)
    gt_nor = cam.get_normal_map()
    direction_cam = gt_nor[x, y, :3]
    direction_cam /= np.linalg.norm(direction_cam)
    direction_world = cam.get_metadata()['mat44'][:3, :3] @ direction_cam
    # print(direction_cam, direction_world)
    out_info['norm_direction_camera' + number] = direction_cam.tolist()
    out_info['norm_direction_world' + number] = direction_world.tolist()

    # The initial direction obeys Gaussian distribution
    degree = np.abs(np.random.normal(loc=0, scale=30, size=[1]))
    radian = degree * np.pi / 180
    threshold = 1 * np.pi / 180
    # sample a random direction in the hemisphere (cam/world)
    up_cam = np.random.randn(3).astype(np.float32)
    up_cam /= np.linalg.norm(up_cam)
    # while action_direction_cam @ direction_cam > -np.cos(np.pi / 6):  # up_norm_thresh: 30
    num_trial = 0
    lower_bound = np.random.uniform(0, 1)**2
    # lower_bound = 0
    while (up_cam @ direction_cam > -np.cos(radian + threshold) or up_cam @ direction_cam < -np.cos(radian - threshold)) and (cam.get_metadata()['mat44'][:3, :3] @ up_cam)[2] > lower_bound \
            and num_trial < 2000 and if_constraint_up:  # up_norm_thresh: 30
        up_cam = np.random.randn(3).astype(np.float32)
        up_cam /= np.linalg.norm(up_cam)
        num_trial += 1
    up = cam.get_metadata()['mat44'][:3, :3] @ up_cam
    if if_given_up == True:
        up = given_up
        up = up / np.linalg.norm(up)
    

    # compute final pose
    out_info['gripper_direction_world' + number] = up.tolist()
    forward = np.random.randn(3).astype(np.float32)
    if if_given_forward == True:
        forward = given_forward
    while (up @ forward) > 0.999:
        forward = np.random.randn(3).astype(np.float32)
    left = np.cross(up, forward)
    left /= np.linalg.norm(left)
    forward = np.cross(left, up)
    forward /= np.linalg.norm(forward)

    if if_given_left == True:
        left = given_left
        while (up @ left) > 0.99:
            left = np.random.randn(3).astype(np.float32)
        forward = np.cross(left, up)
        forward /= np.linalg.norm(forward)
        left = np.cross(up, forward)
        left /= np.linalg.norm(left)
    forward_cam = np.linalg.inv(cam.get_metadata()['mat44'][:3, :3]) @ forward
    out_info['gripper_forward_direction_world' + number] = forward.tolist()
    rotmat = np.eye(4).astype(np.float32)   # rotmat: world coordinate
    rotmat[:3, 0] = forward
    rotmat[:3, 1] = left
    rotmat[:3, 2] = up

    pre_rotmat = np.array(rotmat, dtype=np.float32)
    pre_rotmat[:3, 3] = position_world - up * maneuver_dist
    pre_pose = Pose().from_transformation_matrix(pre_rotmat)

    start_rotmat = np.array(rotmat, dtype=np.float32)
    start_rotmat[:3, 3] = position_world - up * start_dist
    start_pose = Pose().from_transformation_matrix(start_rotmat)
    
    # compute final pose
    forward = start_rotmat[:3, 0]
    left = start_rotmat[:3, 1]
    up = start_rotmat[:3, 2]
    final_rotmat = np.array(start_rotmat, dtype=np.float32)
    final_rotmat[:3, 3] = start_rotmat[:3, 3] + action_direction_world * displacement
    final_pose = Pose().from_transformation_matrix(final_rotmat)
    out_info['pre_rotmat_world' + number] = pre_rotmat.tolist()
    out_info['start_rotmat_world' + number] = start_rotmat.tolist()
    out_info['final_rotmat_world' + number] = final_rotmat.tolist()

    return pre_pose, pre_rotmat, start_pose, start_rotmat, final_pose, final_rotmat, up, forward



def gripper_move_to_target_pose(robot, target_ee_pose, num_steps, vis_gif=False, vis_gif_interval=200, cam=None, monitor=None):
    imgs = []
    # print('target_pose', target_ee_pose)
    executed_time = num_steps * robot.timestep
    spatial_twist = robot.calculate_twist(executed_time, target_ee_pose)

    for i in range(num_steps):
        if i % 100 == 0:
            spatial_twist = robot.calculate_twist((num_steps - i) * robot.timestep, target_ee_pose) # twist at one step
            qvel = robot.compute_joint_velocity_from_twist(spatial_twist)
            # print(i)
            # print(qvel)

        qvel = robot.compute_joint_velocity_from_twist(spatial_twist)
        robot.internal_controller(qvel, i)

        robot.env.step()
        robot.env.render()

        passive_force = robot.robot.compute_passive_force()
        if monitor is not None:
            # monitor.update(variable='pf', new_value_y=passive_force[2])
            monitor.update(variable=f'qpos{0}', new_value_y=robot.robot.get_qpos()[2] - robot.robot.get_drive_target()[2])
            # print(passive_force[5])
        if vis_gif and ((i + 1) % vis_gif_interval == 0):
            rgb_psose, _ = cam.get_observation()
            fimg = (rgb_pose * 255).astype(np.uint8)
            fimg = Image.fromarray(fimg)
            imgs.append(fimg)
        if vis_gif and (i == 0):
            rgb_pose, _ = cam.get_observation()
            fimg = (rgb_pose * 255).astype(np.uint8)
            fimg = Image.fromarray(fimg)
            for idx in range(5):
                imgs.append(fimg)

    if vis_gif:
        return imgs


def gripper_wait_n_steps(robot, n, vis_gif=False, vis_gif_interval=200, cam=None, monitor = None):
    imgs = []

    robot.clear_velocity_command()
    for i in range(n):
        passive_force = robot.robot.compute_passive_force()
        robot.robot.set_qf(passive_force)
        robot.env.step()
        robot.env.render()
        if monitor is not None:
            # monitor.update(variable='pos', new_value_y=robot.robot.get_qpos()[6])
            monitor.update(variable=f'qpos{0}', new_value_y=robot.robot.get_qpos()[2] - robot.robot.get_drive_target()[2])
        if vis_gif and ((i + 1) % vis_gif_interval == 0 or i == 0):
            rgb_pose, _ = cam.get_observation()
            fimg = (rgb_pose * 255).astype(np.uint8)
            fimg = Image.fromarray(fimg)
            imgs.append(fimg)
            if i == 0:
                for _ in range(2):
                    imgs.append(fimg)
    # robot.robot.set_qf([0] * robot.robot.dof)

    if vis_gif:
        return imgs



def append_data_list(file_dir, only_true_data=False, append_root_dir=False, total_dir=False):
    data_list = []
    if file_dir != 'xxx':
        if 'RL' in file_dir:
            data_list.append(os.path.join(file_dir, 'dual_succ_files'))
        else:
            data_list.append(os.path.join(file_dir, 'succ_files'))

        if not only_true_data:
            data_list.append(os.path.join(file_dir, 'fail_files'))
            data_list.append(os.path.join(file_dir, 'invalid_files'))

        if total_dir:
            data_list.append(os.path.join(file_dir, 'total_files'))

        if append_root_dir:
            data_list.append(file_dir)
        print('data_list: ', data_list)
    return data_list


def save_data_full(saved_dir, epoch, out_info, cam_XYZA_list, gt_target_link_mask=None,
              whole_pc=None, repeat_id=None, category=None, shape_id=None, cam_XYZA_list2=None):
    symbol = str(epoch)
    if repeat_id is not None:
        symbol = symbol + "_" + str(repeat_id)
    if category is not None:
        symbol = symbol + "_" + str(category)
    if shape_id is not None:
        symbol = symbol + "_" + str(shape_id)

    with open(os.path.join(saved_dir, 'result_%s.json' % symbol), 'w') as fout:
        json.dump(out_info, fout)

    cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, cam_XYZA = cam_XYZA_list
    save_h5(os.path.join(saved_dir, 'cam_XYZA_%s.h5' % symbol),
            [(cam_XYZA_id1.astype(np.uint64), 'id1', 'uint64'),
             (cam_XYZA_id2.astype(np.uint64), 'id2', 'uint64'),
             (cam_XYZA_pts.astype(np.float32), 'pc', 'float32')])

    if cam_XYZA_list2 is not None:
        cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, cam_XYZA = cam_XYZA_list2
        save_h5(os.path.join(saved_dir, 'cam_XYZA2_%s.h5' % symbol),
                [(cam_XYZA_id1.astype(np.uint64), 'id1', 'uint64'),
                 (cam_XYZA_id2.astype(np.uint64), 'id2', 'uint64'),
                 (cam_XYZA_pts.astype(np.float32), 'pc', 'float32')])

    if whole_pc is not None:
        np.savez(os.path.join(saved_dir, 'collision_visual_shape_%s' % symbol), pts=whole_pc)

    if gt_target_link_mask is not None:
        Image.fromarray((gt_target_link_mask > 0).astype(np.uint8) * 255).save(
            os.path.join(saved_dir, 'interaction_mask_%s.png' % symbol))


def save_data(saved_dir, epoch, out_info, cam_XYZA_list, gt_link_mask, gt_link_mask_after=None, whole_pc=None, repeat_id=None, final_cam_XYZA_list=None):
    if repeat_id == None:
        symbol = str(epoch)
    else:
        symbol = str(epoch) + '_' + str(repeat_id)

    with open(os.path.join(saved_dir, 'result_%s.json' % symbol), 'w') as fout:
        json.dump(out_info, fout)

    cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, cam_XYZA = cam_XYZA_list
    save_h5(os.path.join(saved_dir, 'cam_XYZA_%s.h5' % symbol), [(cam_XYZA_id1.astype(np.uint64), 'id1', 'uint64'),
                                                                (cam_XYZA_id2.astype(np.uint64), 'id2', 'uint64'),
                                                                (cam_XYZA_pts.astype(np.float32), 'pc', 'float32'),
                                                                (cam_XYZA.astype(np.float32), 'xyza', 'float32')])

    if final_cam_XYZA_list != None:
        final_cam_XYZA_id1, final_cam_XYZA_id2, final_cam_XYZA_pts, final_cam_XYZA = final_cam_XYZA_list
        save_h5(os.path.join(saved_dir, 'final_cam_XYZA_%s.h5' % symbol), [(final_cam_XYZA_id1.astype(np.uint64), 'id1', 'uint64'),
                                                                          (final_cam_XYZA_id2.astype(np.uint64), 'id2', 'uint64'),
                                                                          (final_cam_XYZA_pts.astype(np.float32), 'pc', 'float32'),
                                                                          (final_cam_XYZA.astype(np.float32), 'xyza', 'float32')])

    if whole_pc != None:
        np.savez(os.path.join(saved_dir, 'collision_visual_shape_%s' % symbol), pts=whole_pc)

    Image.fromarray((gt_link_mask > 0).astype(np.uint8) * 255).save(
        os.path.join(saved_dir, 'interaction_mask_%s.png' % symbol))
    
    if gt_link_mask_after is not None:
        Image.fromarray((gt_link_mask_after > 0).astype(np.uint8) * 255).save(
            os.path.join(saved_dir, 'final_interaction_mask_%s.png' % symbol))

def save_pre_data(saved_dir, epoch, out_info, cam_XYZA_list, gt_link_mask, gt_link_mask_after=None, whole_pc=None, repeat_id=None, final_cam_XYZA_list=None):
    if repeat_id == None:
        symbol = str(epoch)
    else:
        symbol = str(epoch) + '_' + str(repeat_id)

    with open(os.path.join(saved_dir, 'result_%s.json' % symbol), 'w') as fout:
        json.dump(out_info, fout)

    cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, cam_XYZA = cam_XYZA_list
    save_h5(os.path.join(saved_dir, 'cam_XYZA_%s.h5' % symbol), [(cam_XYZA_id1.astype(np.uint64), 'id1', 'uint64'),
                                                                (cam_XYZA_id2.astype(np.uint64), 'id2', 'uint64'),
                                                                (cam_XYZA_pts.astype(np.float32), 'pc', 'float32'),
                                                                (cam_XYZA.astype(np.float32), 'xyza', 'float32')])

    if final_cam_XYZA_list != None:
        final_cam_XYZA_id1, final_cam_XYZA_id2, final_cam_XYZA_pts, final_cam_XYZA = final_cam_XYZA_list
        save_h5(os.path.join(saved_dir, 'final_cam_XYZA_%s.h5' % symbol), [(final_cam_XYZA_id1.astype(np.uint64), 'id1', 'uint64'),
                                                                          (final_cam_XYZA_id2.astype(np.uint64), 'id2', 'uint64'),
                                                                          (final_cam_XYZA_pts.astype(np.float32), 'pc', 'float32'),
                                                                          (final_cam_XYZA.astype(np.float32), 'xyza', 'float32')])

    if whole_pc != None:
        np.savez(os.path.join(saved_dir, 'collision_visual_shape_%s' % symbol), pts=whole_pc)

    Image.fromarray((gt_link_mask > 0).astype(np.uint8) * 255).save(
        os.path.join(saved_dir, 'interaction_mask_%s.png' % symbol))
    
    if gt_link_mask_after is not None:
        Image.fromarray((gt_link_mask_after > 0).astype(np.uint8) * 255).save(
            os.path.join(saved_dir, 'final_interaction_mask_%s.png' % symbol))


# input sz bszx3x2
# input: bs * 3 * 2
# tensor(forward, up).reshape(-1, 2, 3).permute(0, 2, 1)
def bgs(d6s):
    bsz = d6s.shape[0]
    b1 = F.normalize(d6s[:, :, 0], p=2, dim=1)
    a2 = d6s[:, :, 1]
    b2 = F.normalize(a2 - torch.bmm(b1.view(bsz, 1, -1), a2.view(bsz, -1, 1)).view(bsz, 1) * b1, p=2, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1, b2, b3], dim=1).permute(0, 2, 1)


# batch geodesic loss for rotation matrices
def bgdR(Rgts, Rps):
    Rds = torch.bmm(Rgts.permute(0, 2, 1), Rps)
    Rt = torch.sum(Rds[:, torch.eye(3).bool()], 1)  # batch trace
    # necessary or it might lead to nans and the likes
    theta = torch.clamp(0.5 * (Rt - 1), -1 + 1e-6, 1 - 1e-6)
    return torch.acos(theta)


# 6D-Rot loss
# input sz bszx6
def get_6d_rot_loss(pred_6d, gt_6d):
    # [bug fixed]
    # pred_Rs = self.bgs(pred_6d.reshape(-1, 3, 2))
    # gt_Rs = self.bgs(gt_6d.reshape(-1, 3, 2))
    pred_Rs = bgs(pred_6d.reshape(-1, 2, 3).permute(0, 2, 1))
    gt_Rs = bgs(gt_6d.reshape(-1, 2, 3).permute(0, 2, 1))
    theta = bgdR(gt_Rs, pred_Rs)
    return theta

def simulate_single(env, cam, primact_type, robot, wait_steps, move_steps, pos1_pose, pos2_rotmat, pos3_rotmat):
    result = "VALID"
    gif_imgs = []

    robot.robot.set_root_pose(pos1_pose)
    env.render()

    rgb_pose, _ = cam.get_observation()
    fimg = (rgb_pose * 255).astype(np.uint8)
    fimg = Image.fromarray(fimg)

    try:
        # stage 1
        try:
            imgs = robot.move_to_target_pose(pos2_rotmat, num_steps=move_steps, vis_gif=True, cam=cam)
            gif_imgs.extend(imgs)
            imgs = robot.wait_n_steps(n=wait_steps, vis_gif=True, cam=cam)
            gif_imgs.extend(imgs)
        except ContactError:
            print(f"{robot.robot_name} Single Contact Error when stage1!")
            raise ContactError()

        if 'pushing' in primact_type or 'rotating' in primact_type:
            raise Exception

        # stage 1.5
        try:
            robot.close_gripper()
            robot.wait_n_steps(n=wait_steps, cam=cam)
        except ContactError:
            print(f"{robot.robot_name} Single Contact error when stage1.5!")
            raise ContactError()

        # stage 2: move to start pose
        try:
            imgs = robot.move_to_target_pose(pos3_rotmat, num_steps=move_steps, vis_gif=True, cam=cam)
            gif_imgs.extend(imgs)
            imgs = robot.wait_n_steps(n=wait_steps, vis_gif=True, cam=cam)
            gif_imgs.extend(imgs)
        except ContactError:
            print(f"{robot.robot_name} Single Contact error when stage2!")
            raise ContactError()

    except ContactError:
        result = "INVALID"

    except:
        pass

    return result, fimg, gif_imgs


def cal_reward(primact_type, success, alpha, beta, gamma, traj_len, grip_dir1, grip_dir2, trajectory,
               grasp1, grasp2, next_grasp1, next_grasp2):
    # calculate reward
    reward = 0
    if 'pushing' in primact_type:
        if success:
            reward += 1
        else:
            if np.abs(alpha) < 5 and np.abs(beta) < 5 and np.abs(gamma) < 5 and 0.01 <= traj_len < 0.05:
                reward = traj_len * 10 * 2
            elif (np.abs(alpha) > 5 or np.abs(beta) > 5 or np.abs(gamma) > 5) and traj_len >= 0.01:
                reward = 0.1
            else:
                reward = 0.05  # valid

    elif 'rotating' in primact_type:
        if success:
            reward += 1
        # ......

    elif 'pickup' in primact_type or 'pulling' in primact_type:
        # whether grasp successfully or not
        if (grasp1 and not grasp2) or (not grasp1 and grasp2):
            reward += 0.2
        elif grasp1 and grasp2:
            reward += 0.5
        # whether pick up successfully or not
        if success:
            reward += 1
        else:
            if np.abs(alpha) < 5 and np.abs(beta) < 5 and np.abs(gamma) < 5 and 0.01 <= traj_len < 0.05:
                reward += traj_len * 10 * 2
            elif (np.abs(alpha) > 5 or np.abs(beta) > 5 or np.abs(gamma) > 5) and traj_len >= 0.02:
                reward += 0.2
            else:
                reward += 0.05
        # not single arm
        if next_grasp1 and next_grasp2:
            reward += 0.5
        # cos1 = np.dot(-grip_dir1, trajectory) / np.linalg.norm(grip_dir1) / np.linalg.norm(trajectory)
        # cos2 = np.dot(-grip_dir2, trajectory) / np.linalg.norm(grip_dir2) / np.linalg.norm(trajectory)
        # print("cos1:", cos1, ",cos2:", cos2)
        # reward += 0.3 * (cos1 + cos2)
        if 'pulling' in primact_type:
            cos3 = np.linalg.norm(trajectory[:2]) / np.linalg.norm(trajectory)
            print("cos3:", cos3)
            reward += 0.5 * cos3

    return reward

def check_con(env, cam, cam_XYZA, scene, strict=False):
    cam_pc = cam_XYZA.reshape(cam_XYZA.shape[0] * cam_XYZA.shape[1], -1)
    # print(world_pc)
    object_mask = np.where(cam_pc[:, 3] > 0.6)
    print(object_mask)
    object_pc_cam = cam_pc[object_mask][:, :3]
    # print('object_pc_cam = ', object_pc_cam)
    # print('shape = ', object_pc_cam.shape)
    # print(object_mask)

    assert object_pc_cam.shape[0] == len(object_mask[0]) and object_pc_cam.shape[0] > 0 and object_pc_cam.shape[1] == 3
    # print(object_pc_cam)
    object_pc_world = (cam.mat44 @ np.concatenate((object_pc_cam.T, np.ones((1, object_pc_cam.shape[0]))), axis=0)).T[:,:3]
    # print('obj_pc_world = ', object_pc_world)

    if scene == 'table':
        if np.min(object_pc_world[:, 2]) < 0.8:
            return False
        if strict:
            if np.max(np.abs(object_pc_world[:, 0])) > 0.78 or np.max(np.abs(object_pc_world[:, 1])) > 0.78:
                return True
        else:
            if np.max(np.abs(object_pc_world[:, 0])) > 0.75 or np.max(np.abs(object_pc_world[:, 1])) > 0.75:
                return True
    if scene == 'slope':
        if strict:
            if np.min(object_pc_world[:, 0]) < -0.03:
                return True
        else:
            if np.min(object_pc_world[:, 0]) < 0.:
                return True
    if scene == 'groove':
        if strict:
            if np.min(abs(object_pc_world[:, 0])) < 0.045:
                return True
        else:
            if np.min(abs(object_pc_world[:, 0])) < 0.075:
                return True
    if scene == 'wall':
        if strict:
            if np.min(object_pc_world[:, 0]) < 0.005:
                return True
        else:
            if np.min(object_pc_world[:, 0]) < 0.035:
                return True
    if scene == 'multiple':
        print('start check multiple')
        print(object_pc_world)
        if strict:
            if np.min(object_pc_world[:, 0]) < -0.75:
                return True, 'wall'
            if np.min(object_pc_world[:, 1]) > 0.75+0.03:
                return True, 'slope'
            if np.min(object_pc_world[:, 1]) < -0.75-0.03:
                return True, 'groove'
            if np.min(object_pc_world[:, 0]) > 0.75+0.03:
                return True, 'table'
            else:
                return False, 'none'
        else:
            if np.min(object_pc_world[:, 0]) < -0.75+0.005:
                return True, 'wall'
            if np.min(object_pc_world[:, 1]) > 0.75:
                return True, 'slope'
            if np.min(object_pc_world[:, 1]) < -0.75:
                return True, 'groove'
            if np.min(object_pc_world[:, 0]) > 0.75:
                return True, 'table'
            else:
                return False, 'none'
    return False


def check_success(trajectory, alpha, beta, gamma, task=None, threshold=10, threshold_t=0.05, displacement = 0.4, if_push=None, traj_robot=None, con=False, scene=None):
    success = False

    traj_x = trajectory[0]
    traj_y = trajectory[1]
    traj_z = trajectory[2]
    print("traj_x:", traj_x, ",traj_y:", traj_y, ",traj_z:", traj_z, ",alpha:", alpha, ",beta:", beta, ",gamma:", gamma)
    if if_push == True:
        delta_x = traj_x - traj_robot[0]
        delta_y = traj_y - traj_robot[1]
        delta_z = traj_z - traj_robot[2]
        print('np.abs(traj_x-traj_robot_x[0])', np.abs(delta_x))
        print('np.abs(traj_y-traj_robot_y[1])', np.abs(delta_x))
        if not con:
            return success, traj_x, traj_y, traj_z
        if scene == 'table':
            if np.abs(alpha) < threshold and np.abs(beta) < threshold and np.abs(gamma) < threshold and np.abs(delta_x) < threshold_t and np.abs(delta_y) < threshold_t and np.abs(delta_z) < 0.2 :  # translation but not rotate
                success = True
        elif scene == 'slope':
            scale = 1.5
            scale_t = 1.5
            if np.abs(alpha) < threshold*scale and np.abs(beta) < threshold*scale and np.abs(gamma) < threshold*scale and np.abs(delta_x) < threshold_t*scale_t and np.abs(delta_y) < threshold_t*scale_t:  # translation but not rotate
                success = True
        elif scene == 'groove':
            scale = 100
            scale_t = 1.5
            if np.abs(alpha) < threshold*scale and np.abs(beta) < threshold*scale and np.abs(gamma) < threshold*scale and np.abs(delta_x) < threshold_t*scale_t and np.abs(delta_y) < threshold_t*scale_t:  # translation but not rotate
                success = True
        elif scene == 'wall':
            scale = 1.5
            scale_t = 2
            if np.abs(alpha) < threshold*scale and np.abs(beta) < threshold*scale and np.abs(gamma) < threshold*scale and np.abs(delta_x) < threshold_t*scale_t and np.abs(delta_y) < threshold_t*scale_t:  # translation but not rotate
                success = True
        elif scene == 'multiple':
            scale = 2
            scale_t = 2
            if np.abs(alpha) < threshold*scale and np.abs(beta) < threshold*scale and np.abs(gamma) < threshold*scale and np.abs(delta_x) < threshold_t*scale_t and np.abs(delta_y) < threshold_t*scale_t:
                success = True
        else:
            print('scene error!')
    else:
        if np.abs(alpha) < threshold and np.abs(beta) < threshold and np.abs(gamma) < threshold and traj_z >= displacement - 0.05 and np.abs(traj_x) < threshold_t and np.abs(traj_y) < threshold_t:  # translation but not rotate
            success = True
    
    return success, traj_x, traj_y, traj_z

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


def process_part_pc(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, coordinate_system, mat44=None, cam2cambase=None, object_position_world=None, box_width=1.0, object_mask=None, num_point=8192, device='cuda'):
    pc = cam_XYZA_pts
    #print(mat44)
    if object_position_world is None:
        indices_1D = [x * 448 + y  for x, y in zip(cam_XYZA_id1, cam_XYZA_id2)]
        object_mask = object_mask[indices_1D]
        obj_pc = pc[mask, :]
        object_position_cam = (obj_pc.max(axis=0, keepdims=True) + obj_pc.min(axis=0, keepdims=True)) / 2
        object_position_cambase = object_position_cam @ np.transpose(cam2cambase, (1, 0))
    else:
        object_position_cam = (np.concatenate((object_position_world, np.array([1]))) @ np.transpose(np.linalg.inv(mat44), (1, 0)))[:3]
            #print(object_position_cam)
        object_position_cambase = object_position_cam @ np.transpose(cam2cambase, (1, 0))

    # 搞出来一个pc，在下面处理位置平移和box
    if coordinate_system == 'world':  # cam2world
        pc = (mat44 @ np.concatenate((pc.T, np.ones(len(pc))), axis=0)).T[:, :3]
        mask = (max(np.abs(pc[:, :] - np.tile(object_position_world, (len(pc), 1))), axis=1) < box_width/2)
        pc = pc[mask, :]
        XYZA_id1 = cam_XYZA_id1[mask]
        XYZA_id2 = cam_XYZA_id2[mask]
        pc_centers = (pc.max(axis=0, keepdims=True) + pc.min(axis=0, keepdims=True)) / 2
        pc_centers = pc_centers[0]
        pc -= pc_centers
    elif coordinate_system == 'cambase':  # cam2cambase
        pc = pc @ np.transpose(cam2cambase, (1, 0))
        mask = (np.max(np.abs(pc[:, :] - np.tile(object_position_cambase, (len(pc), 1))), axis=1) < box_width/2)
        pc = pc[mask, :]
        XYZA_id1 = cam_XYZA_id1[mask]
        XYZA_id2 = cam_XYZA_id2[mask]
        pc_centers = (pc.max(axis=0, keepdims=True) + pc.min(axis=0, keepdims=True)) / 2
        pc_centers = pc_centers[0]
        pc -= pc_centers
        # print(pc_centers)
        # print('obj_pose_cambase', object_position_cambase - pc_centers)
    # out = torch.from_numpy(pc).unsqueeze(0)
    indices_1D = [x * 448 + y  for x, y in zip(XYZA_id1, XYZA_id2)]
    object_mask = object_mask[indices_1D]
    XYZA = np.concatenate((pc, object_mask[..., np.newaxis]), axis=1)
    XYZA[object_mask == 0, 3] = 0.5

    #sample
    XYZA = torch.tensor(XYZA).to(device)
    XYZA_id1 = torch.tensor(XYZA_id1).to(device)
    XYZA_id2 = torch.tensor(XYZA_id2).to(device)
    object_pc = XYZA[XYZA[:,3]==1]
    env_pc = XYZA[XYZA[:,3]==0.5]
    objext_id1, objext_id2 = XYZA_id1[XYZA[:,3]==1], XYZA_id1[XYZA[:,3]==1]
    env_id1, env_id2 = XYZA_id1[XYZA[:,3]==0.5], XYZA_id1[XYZA[:,3]==0.5]

    object_num_point = min(len(object_pc), max(int(2*num_point*len(object_pc)/len(XYZA)), 500), 4096)
    env_num_point = num_point - object_num_point
    object_mask = furthest_point_sample(object_pc[:,:3].float().unsqueeze(0), object_num_point).long().reshape(-1)
    env_mask = furthest_point_sample(env_pc[:,:3].float().unsqueeze(0), env_num_point).long().reshape(-1)
    #print(object_pc.shape, object_mask.shape, env_pc.shape, env_mask.shape, object_pc[object_mask].shape)
    if len(object_pc)==0:
        XYZA = env_pc[env_mask].cpu().numpy()
        XYZA_id1 = env_id1[env_mask].cpu().numpy()
        XYZA_id2 = env_id2[env_mask].cpu().numpy()
    elif len(env_pc)==0:
        XYZA = object_pc[object_mask].cpu().numpy()
        XYZA_id1 = objext_id1[object_mask].cpu().numpy()
        XYZA_id2 = objext_id1[object_mask].cpu().numpy()
    else:
        XYZA = torch.cat((object_pc[object_mask],env_pc[env_mask]), axis=0).cpu().numpy()
        XYZA_id1 = torch.cat((objext_id1[object_mask],env_id1[env_mask]), axis=0).cpu().numpy()
        XYZA_id2 = torch.cat((objext_id2[object_mask],env_id2[env_mask]), axis=0).cpu().numpy()
    
    # print(XYZA)
    return XYZA_id1, XYZA_id2, XYZA[:,:3], XYZA, pc_centers

def get_part_pc(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, coordinate_system, mat44=None, cam2cambase=None, gt_target_link_mask=None):
    out = Camera.compute_XYZA_matrix_whole(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, 448, 448) #?可以添加透明度信息
    mask = (out[:, :, 3] > 0.4)
    pc = cam_XYZA_pts
    idx = np.arange(pc.shape[0])
    np.random.shuffle(idx)
    while len(idx) < 30000:
        idx = np.concatenate([idx, idx])
    idx = idx[:30000 - 1]
    pc = pc[idx, :]
    # pc[:, 0] -= 5
    # pc[:, 2] += 2
    pc_centers = None

    if coordinate_system == 'world':  # cam2world
        pc = (mat44 @ np.concatenate((pc.T, np.ones(len(pc))), axis=0)).T[:, :3]
        pc_centers = (pc.max(axis=0, keepdims=True) + pc.min(axis=0, keepdims=True)) / 2
        pc_centers = pc_centers[0]
        pc -= pc_centers
    elif coordinate_system == 'cambase':  # cam2cambase
        pc = pc @ np.transpose(cam2cambase, (1, 0))
        pc_centers = (pc.max(axis=0, keepdims=True) + pc.min(axis=0, keepdims=True)) / 2
        pc_centers = pc_centers[0]
        pc -= pc_centers
    # out = torch.from_numpy(pc).unsqueeze(0)
    return pc, pc_centers

def wait_for_object_still(env, cam=None, visu=False, monitor = None, have_table=False):
    print('start wait for still')
    still_timesteps, wait_timesteps = 0, 0
    imgs = []
    las_qpos = env.get_object_qpos()            
    las_root_pose = env.get_object_root_pose()  
    no_qpos = (las_qpos.shape == (0,))
    while still_timesteps < 200 and wait_timesteps < 10000:
        env.step()
        env.render()
        cur_qpos = env.get_object_qpos()
        cur_root_pose = env.get_object_root_pose()
        if wait_timesteps % 10 == 0:
            if monitor is not None:
        #         monitor.update(variable='p_qf', new_value_y=env.object.compute_passive_force()[2])
        #         monitor.update(variable='qvel', new_value_y=env.object.get_qvel()[2])
                monitor.update(variable='qpos', new_value_y=env.object.get_qpos()[0])
                # print(env.object.get_qpos())
        #     print(cur_root_pose.q)
        #     print("target drive: ", env.object.get_drive_target())
            # print("passive force: ", env.object.compute_passive_force())
            # print("qforce: ", env.object.get_qf())
            # print("qpos: ", env.object.get_qpos())
            # print("qvel: ", env.object.get_qvel())
        #     print("base_pos", env.object.get_base_links()[0].get_pose())
        invalid_contact = False
        for c in env.scene.get_contacts():
            for p in c.points:
                if c.actor1.get_id() not in env.all_link_ids and c.actor2.get_id() not in env.all_link_ids:
                    continue
                if c.actor1.get_id() in env.all_link_ids and c.actor2.get_id() in env.all_link_ids:
                    continue
                if ((c.actor1.get_name() == 'ground' and c.actor2.get_id() in env.all_link_ids) or \
                    (c.actor2.get_name() == 'ground' and c.actor1.get_id() in env.all_link_ids)) and have_table:
                    invalid_contact = True
                    return 0
                if abs(p.impulse @ p.impulse)**0.5 > 5e-3:
                    # print(f"invalid_contact: actor1: {c.actor1.get_name()}, actor2: {c.actor2.get_name()}, wait_timestep: {wait_timesteps}, pulse: {p.impulse}, wait_timesteps: {wait_timesteps}")
                    invalid_contact = True
                    break
            if invalid_contact:
                break
        if (no_qpos or np.max(np.abs(cur_qpos - las_qpos)) < 1e-4) and \
           (np.max(np.abs(cur_root_pose.p - las_root_pose.p)) < 1e-4) and \
           (np.max(np.abs(cur_root_pose.q - las_root_pose.q)) < 1e-4) and (not invalid_contact):
            still_timesteps += 1
        else:
            still_timesteps = 0
        # print(np.max(np.abs(cur_qpos - las_qpos)), np.max(np.abs(cur_root_pose.p - las_root_pose.p)), np.max(np.abs(cur_root_pose.q - las_root_pose.q)))

        las_qpos = cur_qpos
        las_root_pose = cur_root_pose
        wait_timesteps += 1
        if visu and wait_timesteps % 200 == 0:
            rgb_pose, _ = cam.get_observation()
            fimg = (rgb_pose * 255).astype(np.uint8)
            fimg = Image.fromarray(fimg)
            for idx in range(5):
                imgs.append(fimg)
        # if wait_timesteps % 100 == 0:
            # print('still_timesteps: ', still_timesteps, ', wait_timesteps: ', wait_timesteps)
    print('end wait for still, still_timesteps: ', still_timesteps, ', wait_timesteps: ', wait_timesteps)
    if visu:
        return still_timesteps, imgs
    else:
        return still_timesteps


def get_shape_list_full(all_categories, primact, mode='train'):
    tag_dict = {"train": dict(), "val": dict(), "test": dict()}
    if "/" in all_categories:
        cat_list = all_categories.split('/')
    else:
        cat_list = all_categories.split(',')
    shape_cat_dict = dict()
    if primact == "all":
        primacts = ["pushing", "rotating", "pickup"]
    else:
        primacts = [primact]
    for tag in tag_dict:
        for pm in primacts:
            tag_dict[tag]["dir"] = f"../stats/train_where2actPP_{tag}_data_list_{pm}.txt"
            tag_dict[tag]["shape_list"] = list()
            tag_dict[tag]["cat_shape_id_dict"] = dict()
            for cat in cat_list:
                tag_dict[tag]["cat_shape_id_dict"][cat] = list()

            with open(tag_dict[tag]["dir"], 'r') as fin:
                for line in fin.readlines():
                    shape_id, cat = line.rstrip().split()
                    if cat not in cat_list:
                        continue
                    tag_dict[tag]["shape_list"].append(shape_id)
                    tag_dict[tag]["cat_shape_id_dict"][cat].append(shape_id)
                    shape_cat_dict[shape_id] = cat

    if mode == 'all':
        all_shape_list = tag_dict["train"]["shape_list"] + tag_dict["val"]["shape_list"] + tag_dict["test"]["shape_list"]
        all_cat_shape_id_list = dict()
        for cat in cat_list:
            all_cat_shape_id_list[cat] = tag_dict["train"]["cat_shape_id_dict"][cat] + \
                                         tag_dict["val"]["cat_shape_id_dict"][cat] + \
                                         tag_dict["test"]["cat_shape_id_dict"][cat]
        return cat_list, all_shape_list, shape_cat_dict, all_cat_shape_id_list
    else:
        return cat_list, tag_dict[mode]["shape_list"], shape_cat_dict, tag_dict[mode]["cat_shape_id_dict"]


def find_edge_point(xs, ys):
    # Combine xs and ys into a single array of points
    points = np.column_stack((xs, ys))
    
    # Use Nearest Neighbors to find the density of neighbors
    n_neighbors = int(min(100, int(len(points)/8)))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(points)
    distances, _ = nbrs.kneighbors(points)
    
    # The assumption is that edge points have a larger average distance to neighbors
    # Calculate the mean distance for each point
    mean_distances = np.mean(distances, axis=1)
    
    # Normalize the mean distances to get a probability distribution
    # The higher the distance, the higher the probability to be chosen as an edge point
    # probabilities = mean_distances / np.sum(mean_distances)
    # print(mean_distances)
    sorted_indices = np.argsort(mean_distances)
    num_points_chosen = int(0.3*len(sorted_indices))
    # print(num_points_chosen)
    top_10_percent_indices = sorted_indices[-num_points_chosen:]
    
    # Choose a point as an edge point with a probability proportional to its mean distance
    chosen_index = np.random.choice(top_10_percent_indices)
    return points[chosen_index][0], points[chosen_index][1], top_10_percent_indices


def get_shape_list(all_categories, mode='train'):
    train_file_dir = "../stats/train.txt"
    val_file_dir = "../stats/val.txt"
    cat_list = all_categories.split(',')

    train_shape_list, val_shape_list = [], []
    val_cat_shape_id_dict, train_cat_shape_id_dict = {}, {}
    shape_cat_dict = {}

    for cat in cat_list:
        train_cat_shape_id_dict[cat] = []
        val_cat_shape_id_dict[cat] = []

    with open(train_file_dir, 'r') as fin:
        for l in fin.readlines():
            shape_id, cat = l.rstrip().split()
            if cat not in cat_list:
                continue
            train_shape_list.append(shape_id)
            train_cat_shape_id_dict[cat].append(shape_id)
            shape_cat_dict[shape_id] = cat

    with open(val_file_dir, 'r') as fin:
        for l in fin.readlines():
            shape_id, cat = l.rstrip().split()
            if cat not in cat_list:
                continue
            val_shape_list.append(shape_id)
            val_cat_shape_id_dict[cat].append(shape_id)
            shape_cat_dict[shape_id] = cat

    if mode == 'train':
        return cat_list, train_shape_list, shape_cat_dict, train_cat_shape_id_dict
    elif mode == 'val':
        return cat_list, val_shape_list, shape_cat_dict, val_cat_shape_id_dict
    elif mode == 'all':
        all_shape_list = train_shape_list + val_shape_list
        all_cat_shape_id_list = {}
        for cat in cat_list:
            all_cat_shape_id_list[cat] = train_cat_shape_id_dict[cat] + val_cat_shape_id_dict[cat]
        return cat_list, all_shape_list, shape_cat_dict, all_cat_shape_id_list



def draw_affordance_map(fn, pcs, pred_aff_map, ctpt1=None, type='0'):
    print('fn: ', fn)

    if type== '1' or type == '2':
        ctpt1s = []
        for k in range(300):  # jitter
            cur_pt = np.zeros(3)
            cur_pt[0] = ctpt1[0] + np.random.random() * 0.02 - 0.01
            cur_pt[1] = ctpt1[1] + np.random.random() * 0.02 - 0.01
            cur_pt[2] = ctpt1[2] + np.random.random() * 0.02 - 0.01
            ctpt1s.append(cur_pt)
        ctpt1s = np.array(ctpt1s)
        ctpt1s_color = np.ones(300)

    if type == '0':
        render_pts_label_png(fn, pcs, pred_aff_map)
    elif type == '1':
        render_pts_label_png(fn, np.concatenate([pcs, ctpt1s]), np.concatenate([pred_aff_map, ctpt1s_color]))
    else:
        pass


def draw_proposal(fns, pcs, pred_aff_maps, positions, dirs, task_size):  #batched
    # print('fn: ', fns)
    for fn, pc, aff, ctpt, dir in zip(fns, pcs, pred_aff_maps, positions, dirs):
        print(fn)
        if task_size == 6:
            up, forward = dir[0: 3], dir[3: 6]
            ctpts = []
            for k in range(300):  # jitter
                cur_pt = np.zeros(3)
                cur_pt[0] = ctpt[0] + np.random.random() * 0.02 - 0.01
                cur_pt[1] = ctpt[1] + np.random.random() * 0.02 - 0.01
                cur_pt[2] = ctpt[2] + np.random.random() * 0.02 - 0.01
                ctpts.append(cur_pt)
            ctpts = np.array(ctpts)
            ctpts_color = np.ones(300)
            render_proposal_png(fn, np.concatenate([pc, ctpts]), np.concatenate([aff, ctpts_color]), ctpt, up, forward)
        else:
            task = dir
            ctpts = []
            for k in range(300):  # jitter
                cur_pt = np.zeros(3)
                cur_pt[0] = ctpt[0] + np.random.random() * 0.02 - 0.01
                cur_pt[1] = ctpt[1] + np.random.random() * 0.02 - 0.01
                cur_pt[2] = ctpt[2] + np.random.random() * 0.02 - 0.01
                ctpts.append(cur_pt)
            ctpts = np.array(ctpts)
            ctpts_color = np.ones(300)
            render_proposal_png(fn, np.concatenate([pc, ctpts]), np.concatenate([aff, ctpts_color]), ctpt, dir)
        


def coordinate_transform(item, is_pc, transform_type='cambase2world', mat44=None, cam2cambase=None, pc_center=None):
    if transform_type == 'cam2world':
        if is_pc:
            transformed_item = (mat44 @ np.concatenate((item.T, np.array([1])))).T[:3]
        else:
            transformed_item = (mat44[:3, :3] @ item.T)
    elif transform_type == 'world2cam':
        if is_pc:
            transformed_item = (np.linalg.inv(mat44) @ np.concatenate((item.T, np.array([1])))).T[:3]
        else:
            transformed_item = (np.linalg.inv(mat44[:3, :3]) @ item.T)
    elif transform_type == 'cam2cambase':
        transformed_item = item @ np.transpose(cam2cambase, (1, 0))
        if is_pc:
            transformed_item -= pc_center
    elif transform_type == 'cambase2cam':
        if is_pc:
            transformed_item = item + pc_center
        else:
            transformed_item = item
        transformed_item = transformed_item @ np.linalg.inv(np.transpose(cam2cambase, (1, 0)))

    return transformed_item


def batch_coordinate_transform(batch, is_pc, transform_type='cambase2world', mat44=None, cam2cambase=None, pc_center=None):
    if cam2cambase is None:
        cb_up = np.array([0, 0, 1], dtype=np.float32)
        cb_left = np.cross(cb_up, mat44[:3, 0])
        cb_left /= np.linalg.norm(cb_left)
        cb_forward = np.cross(cb_left, cb_up)
        cb_forward /= np.linalg.norm(cb_forward)
        base_mat44 = np.eye(4)
        base_mat44[:3, :3] = np.vstack([cb_forward, cb_left, cb_up]).T
        base_mat44[:3, 3] = mat44[:3, 3]  # cambase2world
        cam2cambase = (np.linalg.inv(base_mat44) @ mat44)[:3, :3]
    transformed_batch = []

    for idx in range(len(batch)):
        transformed_item = coordinate_transform(batch[idx], is_pc[idx], transform_type, mat44, cam2cambase, pc_center)
        transformed_batch.append(transformed_item)
    return transformed_batch


#not used
def get_data_info(cur_data, cur_type='type0', given_task=None):
    cur_dir, shape_id, category, \
    pixel1_idx1, contact_point1, gripper_up1, gripper_forward1, \
    traj, valid, success, epoch, result_idx, mat44, cam2cambase, camera_metadata, joint_angles, pc, pc_center, \
    pixel_ids, target_link_mat44, target_part_trans, transition, \
    contact_point_world1, gripper_up_world1, gripper_forward_world1, task, score_before, score_after = cur_data

    if cur_type == 'type0':     # succ
        pass
    elif cur_type == 'type1':   # succ2fail
        task = given_task
        success = False
    elif cur_type == 'type2':   # fail
        pass
    elif cur_type == 'type3':   # invalid
        task = given_task
    elif cur_type == 'type4':
        pass

    return (cur_dir, shape_id, category,
        pixel1_idx1, contact_point1, gripper_up1, gripper_forward1,
        traj, valid, success, epoch, result_idx, mat44, cam2cambase, camera_metadata, joint_angles, pc, pc_center,
        pixel_ids, target_link_mat44, target_part_trans, transition,
        contact_point_world1, gripper_up_world1, gripper_forward_world1, task, score_before, score_after)
    
    
def select_target_part(env, cam):
    object_all_link_ids = env.all_link_ids
    gt_all_link_mask = cam.get_id_link_mask(object_all_link_ids)  # (448, 448), 0(unmovable) - id(all)
    xs, ys = np.where(gt_all_link_mask > 0)
    
    # to find a link with fixed joint
    target_joint_type = ArticulationJointType.FIX
    tot_trial = 0
    while True:
        idx = np.random.randint(len(xs))
        x, y = xs[idx], ys[idx]
        target_part_id = object_all_link_ids[gt_all_link_mask[x, y] - 1]
        print("id:", target_part_id)
        env.set_target_object_part_actor_id2(target_part_id)
        tot_trial += 1
        if (tot_trial >= 50) or (env.target_object_part_joint_type == target_joint_type):
            break
    if env.target_object_part_joint_type != target_joint_type:
        return None, None, None, None, None, None
    gt_target_link_mask = cam.get_id_link_mask([target_part_id])
    target_pose = env.get_target_part_pose()
    target_link_mat44 = target_pose.to_transformation_matrix()
    prev_origin_world_xyz1 = target_link_mat44 @ np.array([0, 0, 0, 1])
    prev_origin_world = prev_origin_world_xyz1[:3]

    env.render()
    return gt_target_link_mask, prev_origin_world, target_pose, target_link_mat44, x, y


def inference(affordance, actor, critic, pcs, conf, if_pre = False, draw_aff_map = False, draw_proposal_map = False, draw_critic_map = False, out_dir = None, file_id = None, prefix = None):
    batch_size=1
    num_ctpt1 = conf.num_ctpt
    rv1 = conf.rvs_proposal
    task_size = 3 if if_pre else 6
    aff_scores = affordance.inference_whole_pc2(pcs).view(batch_size, conf.num_point_per_shape)  # B * N
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    env_indices = torch.where(pcs[0, :, 3] == 0.5)
    obj_indices = torch.where(pcs[0, :, 3] == 1)
    num_obj_points = len(obj_indices[0])
    aff_scores[0, env_indices[0]] = 0
    if draw_aff_map and out_dir is not None and file_id is not None:
        if task_size == 6:
            aff_scores_draw = 1 / (1 + np.exp(-(aff_scores.clone().cpu().numpy() - 0.5) * 15))
        else:
            aff_scores_draw = 1 / (1 + np.exp(-(2*aff_scores.clone().cpu().numpy() - 0.5) * 15))
        fn = os.path.join(out_dir, f'affordance_map_{file_id}.png')
        if prefix is not None:
            fn = os.path.join(out_dir, f'{prefix}_affordance_map_{file_id}')
        draw_affordance_map(fn, pcs[0].detach().cpu().numpy(), aff_scores_draw,
                                    type='0') # ctpt1=position1[0].detach().cpu().numpy()
    aff_sorted_idx = torch.argsort(aff_scores, dim=1, descending=True).view(batch_size, conf.num_point_per_shape)
    batch_idx = torch.tensor(range(batch_size)).view(batch_size, 1)
    selected_point_num = min(num_obj_points, int(conf.num_point_per_shape * conf.aff_topk))
    if selected_point_num == 0:
        selected_point_num = 1
    selected_idx_idx = torch.randint(0, selected_point_num, size=(batch_size, num_ctpt1))
    selected_idx_aff = aff_sorted_idx[batch_idx, selected_idx_idx]
    position1s = pcs.clone()[batch_idx, selected_idx_aff].view(batch_size * num_ctpt1, -1)[:, :3]
    # pc shape: batchsize*num_per_shape*3
    dir1s = actor.actor_sample_n_diffCtpts(pcs, position1s, rvs_ctpt=num_ctpt1, rvs=rv1).contiguous().view(batch_size * num_ctpt1 * rv1, task_size)
    critic_scores = critic.forward_n_diffCtpts(pcs, position1s, dir1s, rvs_ctpt=num_ctpt1, rvs=rv1).view(batch_size, num_ctpt1 * rv1)
    critic_sorted_idx = torch.argsort(critic_scores, dim=1, descending=True).view(batch_size, num_ctpt1 * rv1)
    selected_idx = critic_sorted_idx[0, :int(conf.critic_topk1)]
    critic_scores = critic_scores[0, selected_idx]
    rvs_draw = 20
    if draw_critic_map and out_dir is not None and file_id is not None and num_obj_points > 0:
        position1s_all = pcs.clone()[batch_idx, obj_indices[0]].view(num_obj_points, -1)[:, :3]
        dir1s_all = torch.zeros(rvs_draw, task_size)
        for i in range(rvs_draw):
            if task_size == 6:
                up = torch.randn(3).float()
                up /= torch.norm(up)
                forward = torch.randn(3).float()
                forward -= torch.dot(up, forward) * up
                forward /= torch.norm(forward)
                dir1s_all[i] = torch.cat([up, forward])
            elif task_size == 3:
                dir1s_all[i] = torch.randn(3).float()
        
        dir1s_all = dir1s_all.repeat(num_obj_points, 1).to(conf.device)

        # infer forward_n_diffCtpts for all points but in several batches
        in_batch_size = 100 # integer multiple of rvs_draw
        num_ctpt = in_batch_size // rvs_draw
        num_batches = int(np.ceil(num_obj_points*rvs_draw / in_batch_size))
        critic_scores_draw_rvs = torch.zeros(num_obj_points*rvs_draw).to(conf.device)
        for i in range(num_batches):
            start_idx = i * in_batch_size
            end_idx = min((i + 1) * in_batch_size, num_obj_points*rvs_draw)
            critic_scores_draw_rvs[start_idx:end_idx] = critic.forward_n_diffCtpts(pcs, position1s_all[i*num_ctpt:min((i+1)*num_ctpt, num_obj_points)], dir1s_all[start_idx:end_idx], rvs_ctpt=(end_idx - start_idx)//rvs_draw, rvs=rvs_draw).view(end_idx - start_idx)
        critic_scores_draw = torch.zeros(len(pcs[0])).to(conf.device)
        critic_scores_draw[obj_indices[0]] = torch.topk(critic_scores_draw_rvs.view(num_obj_points, rvs_draw), 10, dim=1)[0].mean(dim=1)
        fn = os.path.join(out_dir, f'critic_map_{file_id}.png')
        if prefix is not None:
            fn = os.path.join(out_dir, f'{prefix}_critic_map_{file_id}')
        draw_affordance_map(fn, pcs[0].detach().cpu().numpy(), critic_scores_draw.contiguous().cpu().numpy(),
                                    type='0') # ctpt1=position1[0].detach().cpu().numpy()
    positions = position1s[selected_idx // rv1].cpu().numpy()
    directions = dir1s[selected_idx].cpu().numpy()
    if draw_proposal_map:
        fn = os.path.join(out_dir, f'proposal_map_{file_id}')
        if prefix is not None:
            fn = os.path.join(out_dir, f'{prefix}_proposal_map_{file_id}')
        
        pc = pcs.unsqueeze(dim=1).repeat(1, batch_size * int(conf.critic_topk1), 1, 1).reshape(batch_size * int(conf.critic_topk1), conf.num_point_per_shape, -1)[:, :, :3].detach().cpu().numpy()
        fns = []
        for i in range(int(conf.critic_topk1)):
            fn = os.path.join(out_dir, f'{prefix}_proposal_map_{file_id}_{i}')
            fns.append(fn)
        aff_scores_draw_expanded = np.expand_dims(aff_scores_draw, axis=1)
        aff_scores_draw_repeated = np.tile(aff_scores_draw_expanded, (1, int(conf.critic_topk1), 1))
        aff_scores_draw_reshaped = aff_scores_draw_repeated.reshape(batch_size * int(conf.critic_topk1), conf.num_point_per_shape)
        draw_proposal(fns, pc, aff_scores_draw_reshaped, positions, directions, task_size=task_size)
    return positions, directions, aff_scores.view(conf.num_point_per_shape).cpu().numpy(), critic_scores.cpu().numpy()


def get_aff_and_infer(affordance, actor, critic, pcs, conf, if_pre = False):
    batch_size=1
    num_ctpt1 = conf.num_ctpt
    rv1 = conf.rvs_proposal
    task_size = 3 if if_pre else 6
    aff_scores = affordance.inference_whole_pc2(pcs).view(batch_size, conf.num_point_per_shape)  # B * N

    env_indices = torch.where(pcs[0, :, 3] == 0.5)
    obj_indices = torch.where(pcs[0, :, 3] == 1)
    num_obj_points = len(obj_indices[0])
    aff_scores[0, env_indices[0]] = 0

    if task_size == 6:
        aff_scores_draw = (1 / (1 + np.exp(-(aff_scores.clone().cpu().numpy() - 0.5) * 4)))[0]
    else:
        aff_scores_draw = (1 / (1 + np.exp(-(2*aff_scores.clone().cpu().numpy() - 0.5) * 4)))[0]

    aff_sorted_idx = torch.argsort(aff_scores, dim=1, descending=True).view(batch_size, conf.num_point_per_shape)
    batch_idx = torch.tensor(range(batch_size)).view(batch_size, 1)
    selected_point_num = min(num_obj_points, int(conf.num_point_per_shape * conf.aff_topk))
    if selected_point_num == 0:
        selected_point_num = 1
    selected_idx_idx = torch.randint(0, selected_point_num, size=(batch_size, num_ctpt1))
    selected_idx_aff = aff_sorted_idx[batch_idx, selected_idx_idx]
    position1s = pcs.clone()[batch_idx, selected_idx_aff].view(batch_size * num_ctpt1, -1)[:, :3]
    dir1s = actor.actor_sample_n_diffCtpts(pcs, position1s, rvs_ctpt=num_ctpt1, rvs=rv1).contiguous().view(batch_size * num_ctpt1 * rv1, task_size)

    critic_scores = critic.forward_n_diffCtpts(pcs, position1s, dir1s, rvs_ctpt=num_ctpt1, rvs=rv1).view(batch_size, num_ctpt1 * rv1)
    critic_sorted_idx = torch.argsort(critic_scores, dim=1, descending=True).view(batch_size, num_ctpt1 * rv1)
    selected_idx = critic_sorted_idx[0, :int(conf.critic_topk1)]
    critic_scores = critic_scores[0, selected_idx]
    rvs_draw = 20

    position1s_all = pcs.clone()[batch_idx, obj_indices[0]].view(num_obj_points, -1)[:, :3]
    dir1s_all = torch.zeros(rvs_draw, task_size)
    for i in range(rvs_draw):
        if task_size == 6:
            up = torch.randn(3).float()
            up /= torch.norm(up)
            forward = torch.randn(3).float()
            forward -= torch.dot(up, forward) * up
            forward /= torch.norm(forward)
            dir1s_all[i] = torch.cat([up, forward])
        elif task_size == 3:
            dir1s_all[i] = torch.randn(3).float()
    
    dir1s_all = dir1s_all.repeat(num_obj_points, 1).to(conf.device)

    # infer forward_n_diffCtpts for all points but in several batches
    in_batch_size = 100 # integer multiple of rvs_draw
    num_ctpt = in_batch_size // rvs_draw
    num_batches = int(np.ceil(num_obj_points*rvs_draw / in_batch_size))
    critic_scores_draw_rvs = torch.zeros(num_obj_points*rvs_draw).to(conf.device)
    for i in range(num_batches):
        start_idx = i * in_batch_size
        end_idx = min((i + 1) * in_batch_size, num_obj_points*rvs_draw)
        critic_scores_draw_rvs[start_idx:end_idx] = critic.forward_n_diffCtpts(pcs, position1s_all[i*num_ctpt:min((i+1)*num_ctpt, num_obj_points)], dir1s_all[start_idx:end_idx], rvs_ctpt=(end_idx - start_idx)//rvs_draw, rvs=rvs_draw).view(end_idx - start_idx)

    critic_scores_draw = torch.zeros(len(pcs[0])).to(conf.device)
    critic_scores_draw[obj_indices[0]] = torch.topk(critic_scores_draw_rvs.view(num_obj_points, rvs_draw), 10, dim=1)[0].mean(dim=1)
    
    positions = position1s[selected_idx // rv1].cpu().numpy()
    directions = dir1s[selected_idx].cpu().numpy()
    
    return positions, directions, aff_scores_draw