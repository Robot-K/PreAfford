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
parser.add_argument('--out_folder_final', type=str, default='xxx')

parser.add_argument('--start_dist', type=float, default=0.345)
parser.add_argument('--maneuver_dist', type=float, default=0.4)
parser.add_argument('--displacement', type=float, default=0.45)
parser.add_argument('--move_steps', type=int, default=3500)
parser.add_argument('--maneuver_steps', type=int, default=1500)
parser.add_argument('--wait_steps', type=int, default=1000)
parser.add_argument('--density', type=float, default=2.0)
parser.add_argument('--damping', type=int, default=10)
parser.add_argument('--initialize_dict', action='store_true', default=False)

parser.add_argument('--num_processes', type=int, default=1)
parser.add_argument('--no_gui', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--multiple_check', action='store_true', default=False)

args = parser.parse_args()
ctx = torch.multiprocessing.get_context("spawn")

Thin_categories = ['Switch', 'Laptop', 'Remote', 'Scissors', 'Window', 'Keyboard2', 'Pen', 'USB', 'Bowl_lie', 'Cap']
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

def write_json(cat_cnt_dict_path, scene_cnt_dict_path, selected_cat, scene, result):
    if result == 'error':
        return
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

def run_jobs(idx_process, args, transition_Q, cur_file_list, dir):
    device = args.device if torch.cuda.is_available() else "cpu"

    random.seed(datetime.datetime.now())
    setup_seed(random.randint(1, 1000) + idx_process)
    cur_random_seed = random.randint(1, 1000) + idx_process

    for trial_id, file_list in enumerate(cur_file_list):
        file, file_id = file_list
        try:
            with open(file, 'r') as fin:
                result_data = json.load(fin)
        except Exception:
            print('fail to load %s' % file)
            continue
        category, scene = result_data['category'], result_data['scene']
        cmd = 'python eval_manipulate_scene_fir.py --file %s --file_id %d --random_seed %d ' \
              '--density %f --damping %d --start_dist %f --maneuver_dist %f --displacement %f ' \
              '--move_steps %d --maneuver_steps %d --wait_steps %d --out_dir %s --out_dir_final %s' \
              ' --no_gui ' \
              % (file, file_id, cur_random_seed, args.density, args.damping, args.start_dist, args.maneuver_dist, args.displacement,
                 args.move_steps, args.maneuver_steps, args.wait_steps, dir, args.out_folder_final)
        if args.multiple_check:
            cmd += '--multiple_check '
        cmd += '> /dev/null 2>&1'

        ret = call(cmd, shell=True)
        print(f"ret = {ret}", f"seed = {cur_random_seed}")
        if ret == 0:
            result = 'con'
        elif ret == 1:
            result = 'nocon'
        elif ret == 2:
            result = 'invalid'
        else:
            result = 'error'
        if ret == 139:
            with open('139.txt', 'a') as fout:
                fout.write(cmd + '\n')
        if ret == 100:
            with open('100.txt', 'a') as fout:
                fout.write(cmd + '\n')
        if ret == 101:
            with open('101.txt', 'a') as fout:
                fout.write(cmd + '\n')
        if ret == 103:
            with open('103.txt', 'a') as fout:
                fout.write(cmd + '\n')
        transition_Q.put([result, trial_id, scene, category])
        with open('cmd.txt', 'a') as fout:
            fout.write(cmd + '\n')
        
if __name__ == '__main__':
    # if args.use_CA:
    #     out_dir = os.path.join(args.CA_path, args.out_folder)
    # else:
    #     out_dir = os.path.join(args.actor1_path, args.out_folder)
    dir = args.out_folder
    cat_cnt_dict_path = os.path.join(dir, 'cat_cnt_dict_pre.json')
    scene_cnt_dict_path = os.path.join(dir, 'scene_cnt_dict_pre.json')

    print('dir: ', dir)
    Thin_categories = ['Switch', 'Laptop', 'Remote', 'Scissors', 'Window', 'Keyboard2', 'Pen', 'USB', 'Bowl_lie', 'Cap', 'Phone']
    Pickable_categories = ['Box', 'Bucket', 'Display', 'Eyeglasses', 'Faucet', 'Kettle', 'KitchenPot', 'Pliers', 'Basket', 'Bowl']
    categories = args.categories.split(',') if args.categories is not None else Thin_categories + Pickable_categories
    scene_list = args.scene.split(',')

    if args.initialize_dict:
        cat_cnt_dict = dict()
        scene_cnt_dict = dict()
        combined_categories = Thin_categories + Pickable_categories
        for obj in combined_categories:
            cat_cnt_dict[obj] = {'con': 0, 'nocon': 0, 'invalid': 0, 'total': 0}
        for scene in scene_list:
            scene_cnt_dict[scene] = {'con': 0, 'nocon': 0, 'invalid': 0, 'total': 0}
        with open(cat_cnt_dict_path, 'w') as json_file:
            json.dump(cat_cnt_dict, json_file)
        with open(scene_cnt_dict_path, 'w') as json_file:
            json.dump(scene_cnt_dict, json_file)
    else:
        with open(cat_cnt_dict_path, 'r') as json_file:
            cat_cnt_dict = json.load(json_file)
        with open(scene_cnt_dict_path, 'r') as json_file:
            scene_cnt_dict = json.load(json_file)

    if not os.path.exists(dir):
        print('dir not exists')
        exit(0)

    if not os.path.exists(args.out_folder_final):
        os.makedirs(os.path.join(args.out_folder_final, 'succ_gif'))
        os.makedirs(os.path.join(args.out_folder_final, 'fail_gif'))
        os.makedirs(os.path.join(args.out_folder_final, 'invalid_gif'))
        os.makedirs(os.path.join(args.out_folder_final, 'total_gif'))
        # os.makedirs(os.path.join(out_dir, 'tmp_succ_gif'))

        os.makedirs(os.path.join(args.out_folder_final, 'succ_files'))
        os.makedirs(os.path.join(args.out_folder_final, 'fail_files'))
        os.makedirs(os.path.join(args.out_folder_final, 'invalid_files'))
        os.makedirs(os.path.join(args.out_folder_final, 'total_files'))
    
    file_list = []

    dir_name = 'total_files'
    for file in tqdm(sorted(os.listdir(os.path.join(dir, dir_name)))):
        if file[-4:] != 'json':
            continue
        file_id = int(file.split('.')[0][7:])
        try:
            with open(os.path.join(dir, dir_name, file), 'r') as fin:
                result_data = json.load(fin)
        except Exception:
            print('fail to load %s' % os.path.join(dir, dir_name, file))
            continue
        cur_cat, shape_id, cur_scene = result_data['category'], result_data['shape_id'], result_data['scene']
        if args.categories is not None and cur_cat not in categories:
            continue
        file_list.append([os.path.join(dir, dir_name, file), file_id])
    # file_list = [[os.path.join(dir, dir_name, 'result_534099.json'), 534099]]

    num_file_per_process = len(file_list) // args.num_processes + 1

    trans_q = mp.Queue()
    processes = []
    for idx_process in range(args.num_processes):
        cur_file_list = file_list[idx_process * num_file_per_process: min((idx_process + 1) * num_file_per_process, len(file_list))]
        p = mp.Process(target=run_jobs, args=(idx_process, args, trans_q, cur_file_list, dir))
        p.start()
        processes.append(p)

    total, max_trial = 0, 0
    cnt_dict = {'con': 0, 'nocon': 0, 'invalid': 0, 'error': 0}
    # print(args.con_ratio)

    t0 = time.time()
    t_begin = datetime.datetime.now()
    while True:
        if not trans_q.empty():
            result, trial_id, scene, category = trans_q.get()
            cnt_dict[result] += 1
            total += 1
            max_trial = max(max_trial, trial_id)
            write_json(cat_cnt_dict_path, scene_cnt_dict_path, category, scene, result)

            print(
                'Episode: {} | trial_id: {} | Valid_portion: {:.4f} | Succ_portion: {:.4f} | Running Time: {:.4f} | Total Time:'.format(
                    total, max_trial, (cnt_dict['con'] + cnt_dict['nocon']) / total, cnt_dict['con'] / total, time.time() - t0), datetime.datetime.now() - t_begin
            )
            t0 = time.time()