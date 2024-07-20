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
parser.add_argument('--con_ratio', type=float, default=1.0)
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
parser.add_argument('--multiple_check', action='store_true', default=False)

args = parser.parse_args()
ctx = torch.multiprocessing.get_context("spawn")

Thin_categories = ['Switch', 'Laptop', 'Remote', 'Scissors', 'Window', 'Keyboard2', 'Pen', 'USB', 'Bowl_lie', 'Cap', 'Phone']
Pickable_categories = ['Box', 'Bucket', 'Display', 'Eyeglasses', 'Faucet', 'Kettle', 'KitchenPot', 'Pliers', 'Basket', 'Bowl']

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

def run_jobs(idx_process, args, transition_Q, out_dir):
    device = args.device if torch.cuda.is_available() else "cpu"

    random.seed(datetime.datetime.now())
    setup_seed(random.randint(1, 1000) + idx_process)
    sum_trial = args.single_trial
    for trial in range(args.start_epoch, sum_trial):
        cur_trial = sum_trial * idx_process + trial
        cur_random_seed = np.random.randint(10000000)
        try:
            with open(cat_cnt_dict_path, 'r') as json_file:
                cat_cnt_dict = json.load(json_file)
            with open(scene_cnt_dict_path, 'r') as json_file:
                scene_cnt_dict = json.load(json_file)
        except Exception:
            continue

        if random.uniform(0, 1) < args.thin_ratio:
            selected_cat = select_key_from_low_half(cat_cnt_dict, Thin_categories)
        else:
            selected_cat = select_key_from_low_half(cat_cnt_dict, Pickable_categories)
            if selected_cat is None:
                selected_cat = select_key_from_low_half(cat_cnt_dict, Thin_categories)
        
        scene = select_key_from_low_half(scene_cnt_dict, args.scene.split(','))
        shape_id = cat2shape_dict[selected_cat][random.randint(0, len(cat2shape_dict[selected_cat]) - 1)]
        cmd = 'python eval_generate_scene.py --trial_id %d --shape_id %s --category %s --random_seed %d ' \
              '--density %f --damping %d --target_part_state %s --start_dist %f --maneuver_dist %f --displacement %f ' \
              '--move_steps %d --maneuver_steps %d --wait_steps %d --out_dir %s --scene %s ' \
              '--use_4th_feature --aff1_version %s --aff1_path %s --aff1_eval_epoch %s ' \
              '--actor1_version %s --actor1_path %s --actor1_eval_epoch %s ' \
              '--critic1_version %s --critic1_path %s --critic1_eval_epoch %s ' \
              '--aff_topk %f --critic_topk1 %f --num_ctpt1 %d --rv1 %d --num_pair1 %d --num_ctpts %d ' \
              '--z_dim %d --feat_dim %d --task_feat_dim %d --cp_feat_dim %d --dir_feat_dim %d --num_point_per_shape %d --device %s --cat_cnt_dict_path %s --scene_cnt_dict_path %s --no_gui ' \
              % (cur_trial, shape_id, selected_cat, cur_random_seed,
                 args.density, args.damping, args.target_part_state, args.start_dist, args.maneuver_dist, args.displacement,
                 args.move_steps, args.maneuver_steps, args.wait_steps, out_dir, scene, args.aff1_version, args.aff1_path, args.aff1_eval_epoch, args.actor1_version, args.actor1_path, args.actor1_eval_epoch, args.critic1_version, args.critic1_path, args.critic1_eval_epoch, args.aff_topk, args.critic_topk1, args.num_ctpt1, args.rv1, args.num_pair1, args.num_ctpts, args.z_dim, args.feat_dim, args.task_feat_dim, args.cp_feat_dim, args.dir_feat_dim, args.num_point_per_shape, device, cat_cnt_dict_path, scene_cnt_dict_path)
        if trial % args.save_interval == 0:
            cmd += '--save_data '
        if np.random.uniform() < args.con_ratio:
            cmd += '--con '
        if args.multiple_check:
            cmd += '--multiple_check '
        cmd += '> /dev/null 2>&1'

        ret = call(cmd, shell=True)
        print(f"ret = {ret}", f"seed = {cur_random_seed}")
        transition_Q.put([ret, trial])
        if ret == 139:
            with open('cmd_139.txt', 'a') as fout:
                fout.write(cmd + '\n')
        if ret == 6:
            with open('cmd_6.txt', 'a') as fout:
                fout.write(cmd + '\n')
        # if ret == 0:
        #     transition_Q.put(['succ', trial])
        #     result = 'succ'
        # elif ret == 1:
        #     transition_Q.put(['fail', trial])
        #     result = 'fail'
        # else:
        #     transition_Q.put(['invalid', trial])
        #     result = 'invalid'

if __name__ == '__main__':
    # if args.use_CA:
    #     out_dir = os.path.join(args.CA_path, args.out_folder)
    # else:
    #     out_dir = os.path.join(args.actor1_path, args.out_folder)
    out_dir = os.path.join(args.out_folder)
    print('out_dir: ', out_dir)
    cat_cnt_dict_path = out_dir+'/cat_cnt_dict.json'
    scene_cnt_dict_path = out_dir+'/scene_cnt_dict.json'

    if os.path.exists(out_dir):
        response = input('Out directory "%s" already exists, continue? (y/n) ' % out_dir)
        if response != 'y' and response != 'Y':
            sys.exit()

    if not os.path.exists(out_dir):
        os.makedirs(os.path.join(out_dir, 'succ_gif'))
        os.makedirs(os.path.join(out_dir, 'fail_gif'))
        os.makedirs(os.path.join(out_dir, 'invalid_gif'))
        os.makedirs(os.path.join(out_dir, 'total_gif'))

        os.makedirs(os.path.join(out_dir, 'succ_files'))
        os.makedirs(os.path.join(out_dir, 'fail_files'))
        os.makedirs(os.path.join(out_dir, 'invalid_files'))
        os.makedirs(os.path.join(out_dir, 'total_files'))

    Thin_categories = ['Switch', 'Laptop', 'Remote', 'Scissors', 'Window', 'Keyboard2', 'Pen', 'USB', 'Bowl_lie', 'Cap', 'Phone']
    Pickable_categories = ['Box', 'Bucket', 'Display', 'Eyeglasses', 'Faucet', 'Kettle', 'KitchenPot', 'Pliers', 'Basket', 'Bowl']
    # Scene_list = ['table', 'wall', 'groove', 'slope']
    cat_cnt_dict_path = out_dir+'/cat_cnt_dict.json'
    scene_cnt_dict_path = out_dir+'/scene_cnt_dict.json'

    
    cat_list, shape_list, shape2cat_dict, cat2shape_dict = utils.get_shape_list(all_categories=args.categories, mode=args.mode)
    scene_list = args.scene.split(',')

    Thin_categories = [cat for cat in Thin_categories if cat in cat_list]
    Pickable_categories = [cat for cat in Pickable_categories if cat in cat_list]
    if args.initialize_dict:
        cat_cnt_dict = dict()
        scene_cnt_dict = dict()
        combined_categories = Thin_categories + Pickable_categories
        for obj in combined_categories:
            cat_cnt_dict[obj] = {'succ': 0, 'fail': 0, 'invalid': 0, 'total': 0}
        for scene in scene_list:
            scene_cnt_dict[scene] = {'succ': 0, 'fail': 0, 'invalid': 0, 'total': 0}
        with open(cat_cnt_dict_path, 'w') as json_file:
            json.dump(cat_cnt_dict, json_file)
        with open(scene_cnt_dict_path, 'w') as json_file:
            json.dump(scene_cnt_dict, json_file)
    else:
        with open(cat_cnt_dict_path, 'r') as json_file:
            cat_cnt_dict = json.load(json_file)
        with open(scene_cnt_dict_path, 'r') as json_file:
            scene_cnt_dict = json.load(json_file)

    trans_q = mp.Queue()
    processes = []
    for idx_process in range(args.num_processes):
        p = mp.Process(target=run_jobs, args=(idx_process, args, trans_q, out_dir))
        p.start()
        processes.append(p)


    total, max_trial = 0, 0
    total_succ = 0


    t0 = time.time()
    t_begin = datetime.datetime.now()
    while True:
        if not trans_q.empty():
            ret, trial_id = trans_q.get()
            if ret == 0:
                total_succ += 1
            # if succ_cnt > args.num_pair1:
            #     succ_cnt = 0
            #     continue
            total += 1
            max_trial = max(max_trial, trial_id)

            print(
                'Episode: {} | trial_id: {} | Succ_portion: {:.4f} | Running Time: {:.4f} | Total Time:'.format(
                    total, max_trial, total_succ / total, time.time() - t0), datetime.datetime.now() - t_begin
            )
            t0 = time.time()

            if total >= args.total_trial:
                for p in processes:
                    print(f"terminating process {p}")
                    p.terminate()
                    p.join()
                break
