import os
import h5py
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import json
from progressbar import ProgressBar
# from pyquaternion import Quaternion
from camera import Camera
import random
import utils
from utils import get_6d_rot_loss
import copy
from tqdm import tqdm
import ipdb



class SAPIENVisionDataset(data.Dataset):

    def __init__(self, category_types, data_features, buffer_max_num, img_size=224,
                 only_true_data=False, succ_proportion=0.4, fail_proportion=0.7, coordinate_system='world',
                 find_task_for_invalid=True, cat2freq=None,
                 no_true_false_equal=False, use_boxed_pc=False, use_4th_feature=True, model_type='first', expand_ctpt=False):
        self.category_types = category_types
        self.data_features = data_features
        self.buffer_max_num = buffer_max_num
        self.img_size = img_size
        self.only_true_data = only_true_data
        self.succ_proportion = succ_proportion
        self.fail_proportion = fail_proportion
        self.coordinate_system = coordinate_system
        self.find_task_for_invalid = find_task_for_invalid
        self.use_boxed_pc = use_boxed_pc
        self.use_4th_feature = use_4th_feature
        self.model_type = model_type
        self.expand_ctpt = expand_ctpt

        self.dataset = []

        cat_list, shape_list, _, _ = utils.get_shape_list(all_categories=category_types, mode='all')
        self.shape_list = shape_list
        self.category_dict = {'success': {}, 'fail': {}, 'invalid': {}, 'total': {}}
        for cat in cat_list:
            self.category_dict['success'][cat] = 0
            self.category_dict['fail'][cat] = 0
            self.category_dict['invalid'][cat] = 0
            self.category_dict['total'][cat] = 0

        self.freq_dict = dict()
        if cat2freq:
            freqs = [int(x) for x in cat2freq.split(',')]
            for idx in range(len(cat_list)):
                self.freq_dict[cat_list[idx]] = freqs[idx]
        else:
            for idx in range(len(cat_list)):
                self.freq_dict[cat_list[idx]] = 1e4


    def load_data(self, data_list):
        bar = ProgressBar()
        num_pos, num_neg, num_invalid, num_total = 0, 0, 0, 0
        for i in bar(range(len(data_list))):
            cur_dir = data_list[i]
            if not os.path.exists(cur_dir):
                print(cur_dir, 'not found')
                continue
            print('start loading data from %s' % cur_dir)

            for root, dirs, files in os.walk(cur_dir):
                fs = []
                for file in sorted(files):
                    if 'json' not in file:
                        continue
                    fs.append(file)
                for file in tqdm(fs):
                    result_idx = int(file.split('.')[0][7:])                
                    with open(os.path.join(cur_dir, 'result_%d.json' % result_idx), 'r') as fin:
                        try:
                            result_data = json.load(fin)
                        except Exception:
                            continue
                    epoch = result_idx
                    shape_id = int(result_data['shape_id'])
                    category = result_data['category']
                    if str(shape_id) not in self.shape_list:
                        continue
                    if category not in self.category_types:
                        continue

                    valid = False if 'invalid' in cur_dir else True
                    success = True if 'succ' in cur_dir else False

                    if self.only_true_data and (not success):
                        print('only true data, skip')
                        continue
                    if (num_neg >= num_total * (self.fail_proportion - self.succ_proportion)) and valid and (not success):
                        continue
                    if (num_invalid >= num_total * (1 - self.fail_proportion)) and (not valid):
                        continue
                    if num_pos >= self.buffer_max_num and success:
                        continue

                    if success and self.category_dict['success'][category] >= self.freq_dict[category]:
                        continue
                    elif valid and (not success) and self.category_dict['fail'][category] >= self.category_dict['total'][category] * (self.fail_proportion - self.succ_proportion):
                        continue
                    elif (not valid) and self.category_dict['invalid'][category] >= self.category_dict['total'][category] * (1 - self.fail_proportion):
                        continue

                    if 'gripper_direction_world1' not in result_data.keys():
                        continue
                    contact_point_world1 = np.array(result_data['position_world1'], dtype=np.float32)
                    gripper_up_world1 = np.array(result_data['gripper_direction_world1'], dtype=np.float32)
                    gripper_forward_world1 = np.array(result_data['gripper_forward_direction_world1'], dtype=np.float32)
                    
                    camera_metadata = result_data['camera_metadata']
                    mat44 = np.array(camera_metadata['mat44'], dtype=np.float32)
                    cam2cambase = np.array(camera_metadata['cam2cambase'], dtype=np.float32)

                    target_link_mat44, target_part_trans, transition = None, None, None     # not used
                    joint_angles = np.array(result_data['joint_angles'], dtype=np.float32)
                    object_position_world = np.array(result_data['obj_pose_p'], dtype=np.float32)

                    if self.model_type == 'first':
                        task = np.append(np.array(result_data['task'], dtype=np.float32), 0.0)
                        
                        score_before = np.array(result_data['score_before'], dtype=np.float32)
                        score_after = np.array(result_data['score_after'], dtype=np.float32)

                        if valid:
                            traj = np.array(result_data['trajectory'], dtype=np.float32)
                            robot_traj = np.array(result_data['traj_robot'], dtype=np.float32)
                            penalty = np.sum((traj[:2] - robot_traj[:2])**2)**0.5 / 0.15+np.sum((task[:2] - robot_traj[:2])**2)**0.5 / 0.15
                        else:
                            traj = None
                            penalty = 0
                    
                    if self.model_type == 'second':
                            traj = None
                        
                    if self.use_boxed_pc:
                        try:
                            with h5py.File(os.path.join(cur_dir, self.coordinate_system, '_XYZA_boxed_%d.h5' % result_idx), 'r') as fin:
                                pc_center = fin['pc_center'][:].astype(np.float32)
                                XYZA = fin['xyza'][:].astype(np.float32)
                                pc = XYZA
                        except Exception:
                            print('fail to load pcs')
                            continue
                    else:
                        try:
                            with h5py.File(os.path.join(cur_dir, 'cam_XYZA_%d.h5' % result_idx), 'r') as fin:
                                cam_XYZA_id1 = fin['id1'][:].astype(np.int64)
                                cam_XYZA_id2 = fin['id2'][:].astype(np.int64)
                                cam_XYZA_pts = fin['pc'][:].astype(np.float32)
                            pc, pc_center = utils.get_part_pc(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, coordinate_system='cambase', mat44=mat44, cam2cambase=cam2cambase)
                        except Exception:
                            continue

                    if self.coordinate_system == 'world':
                        contact_point1, gripper_up1, gripper_forward1 = \
                            contact_point_world1, gripper_up_world1, gripper_forward_world1
                    elif self.coordinate_system == 'cambase':
                        world_batch = [contact_point_world1.copy(), gripper_up_world1.copy(), gripper_forward_world1.copy()]
                        is_pc = [True, False, False]
                        camera_batch = utils.batch_coordinate_transform(world_batch, is_pc, transform_type='world2cam', mat44=mat44)
                        cambase_batch = utils.batch_coordinate_transform(camera_batch, is_pc, transform_type='cam2cambase', cam2cambase=cam2cambase, pc_center=pc_center)
                        contact_point1, gripper_up1, gripper_forward1 = cambase_batch
                        if self.model_type == 'first':
                            task = utils.coordinate_transform(task, False, transform_type='world2cam', mat44=mat44)
                            task = utils.coordinate_transform(task, False, transform_type='cam2cambase', cam2cambase=cam2cambase)


                    if self.model_type == 'second':
                        cur_data = (cur_dir, shape_id, category,
                                    contact_point1, gripper_up1, gripper_forward1,
                                    traj, valid, success, epoch, result_idx, mat44, cam2cambase, camera_metadata, joint_angles, pc, pc_center,
                                    target_link_mat44, target_part_trans, transition,
                                    contact_point_world1, gripper_up_world1, gripper_forward_world1)
                    else:
                        cur_data = (cur_dir, shape_id, category,
                                    contact_point1, gripper_up1, gripper_forward1,
                                    traj, valid, success, epoch, result_idx, mat44, cam2cambase, camera_metadata, joint_angles, pc, pc_center,
                                    target_link_mat44, target_part_trans, transition,
                                    contact_point_world1, gripper_up_world1, gripper_forward_world1, task, score_before, score_after, penalty)

                    if success:
                        if num_pos < self.buffer_max_num * self.succ_proportion:
                            self.category_dict['success'][category] += 1
                            self.category_dict['total'][category] += 1
                            self.dataset.append(cur_data)
                            num_pos += 1
                            num_total += 1
                        if self.model_type == 'first' and np.random.uniform() < 0.0 and not self.only_true_data:
                            new_task = np.random.normal(0, 0.1, 2)
                            while True:
                                if new_task.dot(task[:2]) / (np.linalg.norm(new_task) * np.linalg.norm(task[:2])) < 0.5:
                                    break
                                new_task = np.random.normal(0, 0.1, 2)
                            task[:2] = new_task

                            cur_data = (cur_dir, shape_id, category,
                                    contact_point1, gripper_up1, gripper_forward1,
                                    traj, valid, success, epoch, result_idx, mat44, cam2cambase, camera_metadata, joint_angles, pc, pc_center,
                                    target_link_mat44, target_part_trans, transition,
                                    contact_point_world1, gripper_up_world1, gripper_forward_world1, task, score_before, score_after, penalty)
                            self.dataset.append(cur_data)
                            self.category_dict['fail'][category] += 1
                            self.category_dict['total'][category] += 1
                            num_neg += 1
                            num_total += 1

                    elif valid and (not success):
                        if num_neg < self.buffer_max_num * (self.fail_proportion - self.succ_proportion):
                            self.category_dict['fail'][category] += 1
                            self.category_dict['total'][category] += 1
                            self.dataset.append(cur_data)
                            num_neg += 1
                            num_total += 1

                    elif not valid:     # contact error, the last file_list, the invalid
                        if num_invalid < self.buffer_max_num * (1 - self.fail_proportion):
                            self.category_dict['invalid'][category] += 1
                            self.category_dict['total'][category] += 1
                            self.dataset.append(cur_data)
                            num_invalid += 1
                            num_total += 1

                    # data augmentation with randomly selected contact points
                    if np.random.uniform() < 1.0 and not self.only_true_data and self.expand_ctpt:
                        for i in range(4):
                            obj_indice = np.where(pc[:, 3] < 0.9)[0]
                            if len(obj_indice) > 0:
                                new_contact_point1 = pc[obj_indice[np.random.randint(len(obj_indice))], :3]
                            else:
                                print('no env point, continue')
                                continue
                            if self.model_type == 'second':
                                cur_data = (cur_dir, shape_id, category,
                                            new_contact_point1, gripper_up1, gripper_forward1,
                                            traj, valid, success, epoch, result_idx, mat44, cam2cambase, camera_metadata, joint_angles, pc, pc_center,
                                            target_link_mat44, target_part_trans, transition,
                                            contact_point_world1, gripper_up_world1, gripper_forward_world1)
                                self.dataset.append(cur_data)

                            else:
                                cur_data = (cur_dir, shape_id, category,
                                    contact_point1, gripper_up1, gripper_forward1,
                                    traj, valid, success, epoch, result_idx, mat44, cam2cambase, camera_metadata, joint_angles, pc, pc_center,
                                    target_link_mat44, target_part_trans, transition,
                                    contact_point_world1, gripper_up_world1, gripper_forward_world1, task, score_before, score_after, penalty)
                                self.dataset.append(cur_data)

            print('positive data: %d; negative data: %d; invalid data: %d; total data: %d' % (num_pos, num_neg, num_invalid, num_total))
        print('positive data: %d; negative data: %d; invalid data: %d; total data: %d' % (num_pos, num_neg, num_invalid, num_total))
        print('category distribution: \nsuccess:', self.category_dict['success'], '\nfail:', self.category_dict['fail'], '\ninvalid:', self.category_dict['invalid'], '\ntotal:', self.category_dict['total'])


    def __str__(self):
        strout = '[SAPIENVisionDataset %d] , img_size: %d\n' % \
                (len(self), self.img_size)
        return strout

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        task = None
        score_before = None
        score_after = None
        if self.model_type == 'second':
            cur_dir, shape_id, category, \
            contact_point1, up1, forward1, \
            traj, valid, success, epoch, result_idx, mat44, cam2cambase, camera_metadata, joint_angles, pc, pc_center, \
            target_link_mat44, target_part_trans, transition, \
            contact_point_world1, gripper_up_world1, gripper_forward_world1 = self.dataset[index]
        else:
            cur_dir, shape_id, category, \
            contact_point1, up1, forward1, \
            traj, valid, success, epoch, result_idx, mat44, cam2cambase, camera_metadata, joint_angles, pc, pc_center, \
            target_link_mat44, target_part_trans, transition, \
            contact_point_world1, gripper_up_world1, gripper_forward_world1, task, score_before, score_after, penalty = self.dataset[index]


        # print(result, is_original)
        data_feats = ()
        for feat in self.data_features:
            if feat == 'img':
                with Image.open(os.path.join(cur_dir, 'rgb.png')) as fimg:
                    out = np.array(fimg.resize((self.img_size, self.img_size)), dtype=np.float32) / 255
                out = torch.from_numpy(out).permute(2, 0, 1).unsqueeze(0)
                data_feats = data_feats + (out,)
             
            elif feat == 'part_pc':
                out = torch.from_numpy(pc).unsqueeze(0)
                data_feats = data_feats + (out,)

            elif feat == 'cur_dir':
                data_feats = data_feats + (cur_dir,)

            elif feat == 'shape_id':
                data_feats = data_feats + (shape_id,)
            
            elif feat == 'category':
                data_feats = data_feats + (category,)

            elif feat == 'trajectory':
                data_feats = data_feats + (traj,)

            elif feat == 'task':
                data_feats = data_feats + (task,)

            elif feat == 'ctpt1':
                ctpt1 = contact_point1
                data_feats = data_feats + (ctpt1,)

            elif feat == 'dir1':
                dir1 = np.concatenate([up1, forward1])
                data_feats = data_feats + (dir1,)

            elif feat == 'target_link_mat44':
                data_feats = data_feats + (target_link_mat44,)

            elif feat == 'target_part_trans':
                data_feats = data_feats + (target_part_trans,)

            elif feat == 'pc_centers':
                data_feats = data_feats + (pc_center,)

            elif feat == 'valid':
                data_feats = data_feats + (valid,)

            elif feat == 'success':
                data_feats = data_feats + (success,)

            elif feat == 'result_idx':   # epoch = result_idx
                data_feats = data_feats + (result_idx,)

            elif feat == 'camera_metadata':
                data_feats = data_feats + (camera_metadata,)

            elif feat == 'joint_angles':
                data_feats = data_feats + (joint_angles,)

            # elif feat == 'pixel_ids':
            #     data_feats = data_feats + (pixel_ids,)

            elif feat == 'cam2cambase':
                data_feats = data_feats + (cam2cambase,)

            elif feat == 'mat44':
                data_feats = data_feats + (mat44,)

            elif feat == 'score_after':
                data_feats = data_feats + (score_after,)

            elif feat == 'score_before':
                data_feats = data_feats + (score_before,)

            elif feat == 'penalty':
                data_feats = data_feats + (penalty,)


            else:
                raise ValueError('ERROR: unknown feat type %s!' % feat)

        return data_feats

