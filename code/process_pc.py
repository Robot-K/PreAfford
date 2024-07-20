import os
import h5py
# import torch
# import torch.utils.data as data
import numpy as np
from PIL import Image
import json
from progressbar import ProgressBar
# from pyquaternion import Quaternion
import utils
from utils import get_6d_rot_loss, save_h5
from tqdm import tqdm
from argparse import ArgumentParser
from time import strftime
from data import SAPIENVisionDataset
from pointnet2_ops.pointnet2_utils import furthest_point_sample
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

parser = ArgumentParser()
parser.add_argument('--box_width', type=float, default=1)
parser.add_argument('--num_point_per_shape', type=int, default=8192)
parser.add_argument('--min_num_point_object', type=int, default=1000)
parser.add_argument('--coordinate_system', type=str, default='cambase')
parser.add_argument('--offline_data_dir', type=str, help='data directory')
parser.add_argument('--offline_data_dir2', type=str, default='xxx', help='data directory')
parser.add_argument('--offline_data_dir3', type=str, default='xxx', help='data directory')
parser.add_argument('--offline_data_dir4', type=str, default='xxx', help='data directory')
parser.add_argument('--offline_data_dir5', type=str, default='xxx', help='data directory')
parser.add_argument('--offline_data_dir6', type=str, default='xxx', help='data directory')
parser.add_argument('--save_picture', action='store_true', help='save picture')
parser.add_argument('--save_data', action='store_true', help='save data')
parser.add_argument('--if_final', action='store_true', help='if final')
parser.add_argument('--if_pre_infer', action='store_true', help='if final')
parser.add_argument('--if_total', action='store_true', help='if total')
conf = parser.parse_args()

process_data_list = []
offline_data_dir_list = [conf.offline_data_dir, conf.offline_data_dir2, conf.offline_data_dir3, conf.offline_data_dir4, conf.offline_data_dir5, conf.offline_data_dir6]
for data_dir in offline_data_dir_list:
    # process_data_list.extend([os.path.join(data_dir, 'invalid_files')])
    process_data_list.extend(utils.append_data_list(data_dir, total_dir=conf.if_total))
print(process_data_list)
# process_data_list = ['../data/PreGrasp_train_6/fail_files']

def progress_file(cur_dir, file):
    if 'invalid' in cur_dir and conf.if_final:
        return
    result_idx = int(file.split('.')[0][7:])
    # print(file.split('_'))
    # result_idx = int(file.split('_')[1])
    # print(os.path.join(cur_dir, 'result_%d.json' % result_idx))#testing

    with open(os.path.join(cur_dir, 'result_%d.json' % (result_idx)), 'r') as fin:
        try:
            result_data = json.load(fin)
        except Exception:
            print('Failed to load: %s' % os.path.join(cur_dir, 'result_%d.json' % result_idx))
            return
    
    if not conf.if_total:
        object_position_world = np.array(result_data['obj_pose_p'], dtype=np.float32) # should be changed
    else:
        try:
            object_position_world = np.array(result_data['obj_root_pose_p'], dtype=np.float32)
        except Exception as e:
            print('Failed to load: %s' % os.path.join(cur_dir, 'result_%d.json' % result_idx), e)
    if conf.if_final:
        object_position_world = np.array(result_data['next_obj_pose_p'], dtype=np.float32)
    
    camera_metadata = result_data['camera_metadata']
    mat44 = np.array(camera_metadata['mat44'], dtype=np.float32)
    cam2cambase = np.array(camera_metadata['cam2cambase'], dtype=np.float32)
    
    try:
        if not conf.if_final:
            with h5py.File(os.path.join(cur_dir, 'cam_XYZA_%d.h5' % (result_idx)), 'r') as fin:
                cam_XYZA_id1 = fin['id1'][:].astype(np.int64)
                cam_XYZA_id2 = fin['id2'][:].astype(np.int64)
                cam_XYZA_pts = fin['pc'][:].astype(np.float32)
                # cam_XYZA = fin['xyza'][:,:].astype(np.float32)
        else:
            with h5py.File(os.path.join(cur_dir, 'final_cam_XYZA_%d.h5' % (result_idx)), 'r') as fin:
                cam_XYZA_id1 = fin['id1'][:].astype(np.int64)
                cam_XYZA_id2 = fin['id2'][:].astype(np.int64)
                cam_XYZA_pts = fin['pc'][:].astype(np.float32)
                # cam_XYZA = fin['xyza'][:,:].astype(np.float32)
    except Exception as e:
        print('Failed to load: %s' % os.path.join(cur_dir, 'cam_XYZA_%d.h5' % result_idx), e)
        return

    try:
        #读取interaction_mask_idx.png文件，找到里面有亮度的点。其具体内容参考utils.save_data()函数
        if not conf.if_final:
            image_path = os.path.join(cur_dir, 'interaction_mask_%d.png' % (result_idx))
        else:
            image_path = os.path.join(cur_dir, 'final_interaction_mask_%d.png' % (result_idx)) 
        image = Image.open(image_path).convert('L')
        image_array = np.array(image)
        object_mask = (image_array > 0).astype(int).flatten()
        # print(object_mask)
    except Exception as e:
        print('Failed to find: %s' % os.path.join(cur_dir, 'final_interaction_mask_%s.png' % result_idx), e)
        return

    try:
        XYZA_id1, XYZA_id2, XYZA_pts, XYZA, pc_center = utils.process_part_pc(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, coordinate_system=conf.coordinate_system, mat44=mat44, cam2cambase=cam2cambase, 
                                        object_position_world=object_position_world, box_width=conf.box_width, object_mask=object_mask, num_point=conf.num_point_per_shape)
        if len(XYZA)!=conf.num_point_per_shape:
            return
        
        if conf.save_picture:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            # print(pc[:, 0].numpy())
            object_pc = XYZA[XYZA[:,3]==1]
            env_pc = XYZA[XYZA[:,3]!=1]
            ax.scatter(object_pc[:, 0], object_pc[:, 1], object_pc[:, 2], color='red', s=1, label='object')
            ax.scatter(env_pc[:, 0], env_pc[:, 1], env_pc[:, 2], color='blue', s=1, label='environment')
            # plt.show()
            plt.savefig(os.path.join(cur_dir, 'pcs', f'pcs_%s.png' % result_idx), dpi=600)
            plt.clf()

        if conf.save_data:
            if not conf.if_final and not conf.if_pre_infer:
                save_h5(os.path.join(cur_dir, conf.coordinate_system, '_XYZA_boxed_%s.h5' % result_idx), [(XYZA_id1.astype(np.uint64), 'id1', 'uint64'),
                                                                (XYZA_id2.astype(np.uint64), 'id2', 'uint64'),
                                                                (XYZA_pts.astype(np.float32), 'pc', 'float32'),
                                                                (XYZA.astype(np.float32), 'xyza', 'float32'),
                                                                (pc_center.astype(np.float32), 'pc_center', 'float32')])
            if conf.if_pre_infer:
                save_h5(os.path.join(cur_dir, conf.coordinate_system, 'pre_XYZA_boxed_%s.h5' % result_idx), [(XYZA_id1.astype(np.uint64), 'id1', 'uint64'),
                                                                (XYZA_id2.astype(np.uint64), 'id2', 'uint64'),
                                                                (XYZA_pts.astype(np.float32), 'pc', 'float32'),
                                                                (XYZA.astype(np.float32), 'xyza', 'float32'),
                                                                (pc_center.astype(np.float32), 'pc_center', 'float32')])
            if conf.if_final:
                save_h5(os.path.join(cur_dir, conf.coordinate_system, 'final_XYZA_boxed_%s.h5' % result_idx), [(XYZA_id1.astype(np.uint64), 'id1', 'uint64'),
                                                                (XYZA_id2.astype(np.uint64), 'id2', 'uint64'),
                                                                (XYZA_pts.astype(np.float32), 'pc', 'float32'),
                                                                (XYZA.astype(np.float32), 'xyza', 'float32'),
                                                                (pc_center.astype(np.float32), 'pc_center', 'float32')])
    except Exception as e:
        print('At result_idx%d' % result_idx, e)


bar = ProgressBar()
for i in bar(range(len(process_data_list))):
    cur_dir = process_data_list[i]
    if not os.path.exists(cur_dir):
        print('Not exist: %s' % cur_dir)
        continue
    if not os.path.exists(os.path.join(cur_dir, 'pcs')):
        os.mkdir(os.path.join(cur_dir, 'pcs'))
    if not os.path.exists(os.path.join(cur_dir, conf.coordinate_system)):
        os.mkdir(os.path.join(cur_dir, conf.coordinate_system))
    for root, dirs, files in os.walk(cur_dir):
        fs = []
        for file in sorted(files):
            if 'json' not in file:
                continue
            fs.append(file)
        
        with ThreadPoolExecutor(max_workers=6) as executor:
        # 使用tqdm来创建一个进度条
        # executor.map将progress_file函数应用于files列表中的每个元素
            list(tqdm(executor.map(lambda file: progress_file(cur_dir, file), fs), total=len(fs)))