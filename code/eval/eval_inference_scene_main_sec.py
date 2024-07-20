import os
import sys
import numpy as np
from PIL import Image
import utils
import json
from argparse import ArgumentParser
import torch
import multiprocessing as mp
import time
import random
import imageio
from subprocess import call
import subprocess
# from pointnet2_ops.pointnet2_utils import furthest_point_sample
import math
import datetime
import h5py
from tqdm import tqdm


parser = ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--categories', type=str, help='list all categories [Default: None, meaning all 10 categories]', default=None)
parser.add_argument('--scene', type=str, help='list all scenes [Default: None, meaning all 4 scenes]', default=None)
parser.add_argument('--out_folder', type=str, default='xxx')
parser.add_argument('--repeat_num', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=1)

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

parser.add_argument('--CA_path', type=str)
parser.add_argument('--CA_eval_epoch', type=str)

parser.add_argument('--aff_topk', type=float, default=0.1)
parser.add_argument('--critic_topk1', type=float, default=0.01)
parser.add_argument('--num_ctpt1', type=int, default=10)
parser.add_argument('--rv1', type=int, default=100)
parser.add_argument('--num_pair1', type=int, default=10)
parser.add_argument('--num_ctpts', type=int, default=10)

parser.add_argument('--target_part_state', type=str, default='random-middle')
parser.add_argument('--start_dist', type=float, default=0.345)
parser.add_argument('--maneuver_dist', type=float, default=0.4)
parser.add_argument('--displacement', type=float, default=0.45)
parser.add_argument('--move_steps', type=int, default=3500)
parser.add_argument('--maneuver_steps', type=int, default=1500)
parser.add_argument('--con', action='store_true', default=False)
parser.add_argument('--wait_steps', type=int, default=1000)
parser.add_argument('--density', type=float, default=2.0)
parser.add_argument('--damping', type=int, default=10)

parser.add_argument('--draw_aff_map', action='store_true', default=False)
parser.add_argument('--draw_proposal', action='store_true', default=False)
parser.add_argument('--num_draw', type=int, default=1)

parser.add_argument('--num_processes', type=int, default=1)
parser.add_argument('--single_trial', type=int, default=10000)
parser.add_argument('--total_trial', type=int, default=50000)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--save_interval', type=int, default=5)
parser.add_argument('--thin_ratio', type=float, default=0.8)
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--initialize_dict', action='store_true', default=False)
parser.add_argument('--use_CA', action='store_true', default=False)
parser.add_argument('--no_gui', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--make_dir', action='store_true', default=False)

args = parser.parse_args()
ctx = torch.multiprocessing.get_context("spawn")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def select_key_from_low_half(d, res=None):
    # 如果提供了 res，则只考虑 res 中的键
    if res is not None:
        d = {k: v for k, v in d.items() if k in res}
    # 计算中位数
    if not d:
        # 如果字典为空，可以选择返回None或其他默认值
        return None
    d_sum = {k: sum(v.values()) for k, v in d.items()}
    median_value = sorted(d_sum.values())[len(d_sum) // 2]
    # 找出所有值小于或等于中位数的键
    keys_in_low_half = [k for k, v in d_sum.items() if v <= median_value]
    # 随机选择一个键并返回
    return random.choice(keys_in_low_half) if keys_in_low_half else None

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

def run_jobs(idx_process, args, transition_Q, cur_file_list, out_dir):
    device = args.device if torch.cuda.is_available() else "cpu"

    random.seed(datetime.datetime.now())
    setup_seed(random.randint(1, 1000) + idx_process)

    # load models
    aff1_def = utils.get_model_module(args.aff1_version)
    affordance1 = aff1_def.Network(args.feat_dim, args.cp_feat_dim, args.dir_feat_dim)
    actor1_def = utils.get_model_module(args.actor1_version)
    actor1 = actor1_def.Network(args.feat_dim, args.cp_feat_dim, args.dir_feat_dim, z_dim=args.z_dim)
    critic1_def = utils.get_model_module(args.critic1_version)
    critic1 = critic1_def.Network(args.feat_dim, args.cp_feat_dim, args.dir_feat_dim)
    print_gpu()

    if not args.use_CA:
        print('loading models')
        affordance1.load_state_dict(torch.load(os.path.join(args.aff1_path, 'ckpts', '%s-network.pth' % args.aff1_eval_epoch)))
        print('affordance1 loaded')
        actor1.load_state_dict(torch.load(os.path.join(args.actor1_path, 'ckpts', '%s-network.pth' % args.actor1_eval_epoch)))
        print('actor1 loaded')
        critic1.load_state_dict(torch.load(os.path.join(args.critic1_path, 'ckpts', '%s-network.pth' % args.critic1_eval_epoch)))
        print('critic1 loaded')
    else:
        affordance1.load_state_dict(torch.load(os.path.join(args.CA_path, 'ckpts', '%s-affordance1.pth' % args.CA_eval_epoch)))
        actor1.load_state_dict(torch.load(os.path.join(args.CA_path, 'ckpts', '%s-actor1.pth' % args.CA_eval_epoch)))
        critic1.load_state_dict(torch.load(os.path.join(args.CA_path, 'ckpts', '%s-critic1.pth' % args.CA_eval_epoch)))

    affordance1.to(device).eval()
    actor1.to(device).eval()
    critic1.to(device).eval()
    # print('model to device')
    print_gpu()
    

    for file in cur_file_list:
        # print(file)
        batch_size = args.batch_size

        file_dir, file_id = file
        with open(file_dir, 'r') as fin:
            result_data = json.load(fin)

        try:
            with h5py.File(os.path.join(out_dir, 'cambase', '_XYZA_boxed_%d.h5' % file_id), 'r') as fin:
                XYZA_id1 = fin['id1'][:].astype(np.int64)
                XYZA_id2 = fin['id2'][:].astype(np.int64)
                pc = fin['pc'][:].astype(np.float32)
                # print(pc.shape)
                pc_center = fin['pc_center'][:].astype(np.float32)
                XYZA = fin['xyza'][:].astype(np.float32)
                if args.use_4th_feature:
                    pc = XYZA
        except Exception:
            print('fail to load %s' % os.path.join(out_dir, 'cambase', '_XYZA_boxed_%d.h5' % file_id))
            continue
    
        # print(object_mask)
        pc = torch.from_numpy(pc).unsqueeze(0).to(device)
        shape_id = int(result_data['shape_id'])
        category = result_data['category']
        scene = result_data['scene']
        camera_metadata = result_data['camera_metadata']
        mat44 = np.array(camera_metadata['mat44'], dtype=np.float32)
        cam2cambase = np.array(camera_metadata['cam2cambase'], dtype=np.float32)

        ''' inference '''
        num_ctpt1, rv1 = args.num_ctpt1, args.rv1
        num_pair1 = args.num_pair1


        # print('aff1')
        # print_gpu()

        with torch.no_grad():
            batch_size = 1
            num_pair1 = args.num_pair1
            num_ctpt1, rv1 = args.num_ctpt1, args.rv1
            pc = torch.from_numpy(XYZA).unsqueeze(0).float().to(device)
            aff_scores = affordance1.inference_whole_pc2(pc).view(batch_size, args.num_point_per_shape)  # B * N
            aff_scores[0, np.where(XYZA[:,3]==0.5)] = 0
            # aff_scores = 1 / (1 + np.exp(-(aff_scores.detach().cpu().numpy().reshape(-1) - 0.5) * 15))
            # fn = os.path.join(args.out_dir, 'affordance', '%d_%s_%s_%s' % (args.trial_id, category, shape_id, 'map1'))
            # utils.draw_affordance_map(fn, cam.cam2cambase, cam.mat44, pc[0].detach().cpu().numpy(), aff_scores, type='0')
            aff_sorted_idx = torch.argsort(aff_scores, dim=1, descending=True).view(batch_size, args.num_point_per_shape)
            batch_idx = torch.tensor(range(batch_size)).view(batch_size, 1)
            selected_posi_idx_idx = torch.randint(0, int(args.num_point_per_shape * args.aff_topk), size=(batch_size, num_ctpt1))
            selected_posi_idx = aff_sorted_idx[batch_idx, selected_posi_idx_idx]
            position1s = pc.clone()[batch_idx, selected_posi_idx].reshape(batch_size * num_ctpt1, -1)[:, :3]
            # pc shape: batchsize*num_per_shape*3
            # print('actor1')
            dir1s = actor1.actor_sample_n_diffCtpts(pc, position1s, rvs_ctpt=num_ctpt1, rvs=rv1).contiguous().view(batch_size * num_ctpt1 * rv1, 6)

            # print('critic1')
            critic_scores = critic1.forward_n_diffCtpts(pc, position1s, dir1s, rvs_ctpt=num_ctpt1, rvs=rv1).view(batch_size, num_ctpt1 * rv1)
            critic_sorted_idx = torch.argsort(critic_scores, dim=1, descending=True).view(batch_size, num_ctpt1 * rv1)
            batch_idx = torch.tensor(range(batch_size)).view(batch_size, 1)
            selected_idx_idx = torch.randint(0, int(num_ctpt1 * rv1 * args.critic_topk1), size=(batch_size, num_pair1))
            selected_idx = critic_sorted_idx[batch_idx, selected_idx_idx]
            position1s = position1s.view(batch_size, num_ctpt1, 3)[batch_idx, selected_idx // rv1].view(batch_size * num_pair1, 3)
            dir1s = dir1s.view(batch_size, num_ctpt1 * rv1, 6)[batch_idx, selected_idx].view(batch_size * num_pair1, 6)
            pixel1_idx = selected_posi_idx[batch_idx, selected_idx // rv1]
            # print('start empty inference')
            position1s = position1s.detach().cpu().numpy()
            dir1s = dir1s.detach().cpu().numpy()

        transition_Q.put(['infer_succ', file_id, idx_process])
        
        position1s_world = []
        dir1s_world = []
        for i in range(len(position1s)):
            cambase_batch = [position1s[i].reshape(3), dir1s[i][:3].reshape(3), dir1s[i][3:6].reshape(3)]
            is_pc = [True, False, False]
            camera_batch = utils.batch_coordinate_transform(cambase_batch, is_pc, transform_type='cambase2cam', cam2cambase=cam2cambase, pc_center=pc_center)
            world_batch = utils.batch_coordinate_transform(camera_batch, is_pc, transform_type='cam2world', mat44=mat44)
            position_world1, up_world1, forward_world1 = world_batch
            position1s_world.append(position_world1.tolist())
            dir1s_world.append([up_world1.tolist(), forward_world1.tolist()])
        result_data['position1s'] = position1s_world
        result_data['dir1s'] = dir1s_world
        with open(os.path.join(out_dir, 'inferred', 'result_%d.json' % (file_id)), 'w') as fout:
            json.dump(result_data, fout)
  
        transition_Q.put(['save_succ', file_id, idx_process])


if __name__ == '__main__':
    out_dir = args.out_folder

    print('out_dir: ', out_dir)
    Thin_categories = ['Switch', 'Laptop', 'Remote', 'Scissors', 'Window', 'Keyboard2', 'Pen', 'USB', 'Bowl_lie', 'Cap', 'Phone']
    Pickable_categories = ['Box', 'Bucket', 'Display', 'Eyeglasses', 'Faucet', 'Kettle', 'KitchenPot', 'Pliers', 'Basket', 'Bowl']

    dir_name = 'total_files'
    dir_list = [out_dir]
    for cur_dir in dir_list:
        if not os.path.exists(cur_dir):
            continue
        cur_child_dir = cur_dir.split('/')[-1]
        if not os.path.exists(os.path.join(out_dir, cur_child_dir)) or args.make_dir:
            os.makedirs(os.path.join(out_dir, cur_child_dir, 'affordance_maps'))
            os.makedirs(os.path.join(out_dir, cur_child_dir, 'proposal_maps'))
            os.makedirs(os.path.join(out_dir, cur_child_dir, 'critic_maps'))

    if not os.path.exists(os.path.join(out_dir, dir_name, 'inferred')):
        os.makedirs(os.path.join(out_dir, dir_name, 'inferred'))

    file_list = []

    for cur_dir in dir_list:
        if not os.path.exists(cur_dir):
            continue
        for file in tqdm(sorted(os.listdir(os.path.join(cur_dir, dir_name)))):
            if file[-4:] != 'json':
                continue
            file_id = int(file.split('.')[0][7:])
            # print(os.path.join(cur_dir, dir_name, file))
            try:
                with open(os.path.join(cur_dir, dir_name, file), 'r') as fin:
                    result_data = json.load(fin)
            except Exception:
                print('fail to load %s' % os.path.join(cur_dir, dir_name, file))
                continue
            cur_cat, shape_id, cur_scene = result_data['category'], result_data['shape_id'], result_data['scene']
            file_list.append([os.path.join(cur_dir, dir_name, file), file_id])
            
    num_file_per_process = len(file_list) // args.num_processes + 1

    trans_q = ctx.Queue()
    for idx_process in range(args.num_processes):
        cur_file_list = file_list[idx_process * num_file_per_process: min((idx_process + 1) * num_file_per_process, len(file_list))]
        p = ctx.Process(target=run_jobs, args=(idx_process, args, trans_q, cur_file_list, os.path.join(out_dir, dir_name)))
        p.start()


    total = 0
    infer_succ = 0
    save_succ = 0

    t0 = time.time()
    t_begin = datetime.datetime.now()
    while True:
        if not trans_q.empty():
            results = trans_q.get()
            result, file_id, idx_process = results
            if result == 'infer_succ':
                total += 1
                infer_succ += 1
            if result == 'save_succ':
                save_succ += 1

            print(
                'Episode: {} | Infer_succ: {:.4f} | Save_succ: {:.4f} | Running Time: {:.4f} | Total Time:'.format(
                    total, infer_succ / total, save_succ / total, time.time() - t0), datetime.datetime.now() - t_begin
            )
            t0 = time.time()
