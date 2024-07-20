import os
import sys
import numpy as np
import utils
from argparse import ArgumentParser
import time
import random
import multiprocessing as mp
from subprocess import call
import datetime
import json

parser = ArgumentParser()
parser.add_argument('--category', type=str)
parser.add_argument('--scene', type=str)
parser.add_argument('--out_dir', type=str)
parser.add_argument('--mode', type=str, default='train/val/all')
parser.add_argument('--density', type=float, default=2.0)
parser.add_argument('--damping', type=int, default=10)
parser.add_argument('--target_part_state', type=str, default='random-middle')
parser.add_argument('--start_dist', type=float, default=0.345)
parser.add_argument('--maneuver_dist', type=float, default=0.4)
parser.add_argument('--displacement', type=float, default=0.45)
parser.add_argument('--move_steps', type=int, default=3500)
parser.add_argument('--maneuver_steps', type=int, default=1500)
parser.add_argument('--wait_steps', type=int, default=1000)

parser.add_argument('--single_trail', type=int, default=10000)
parser.add_argument('--total_trail', type=int, default=50000)
parser.add_argument('--num_processes', type=int, default=1)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--save_interval', type=int, default=5)
parser.add_argument('--no_gui', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--use_edge', action='store_true', default=False)
parser.add_argument('--max_num', type=int, default=150)
parser.add_argument('--initialize_dict', action='store_true', default=False)
parser.add_argument('--con_ratio', type=float, default=0.5)
args = parser.parse_args()

cnt_dict_path = args.out_dir+'/cnt_dict_plus.json'
Thin_categories = ['Keyboard2', 'Laptop', 'Scissors', 'Cap', 'Phone']
Pickable_categories = ['Bucket', 'Eyeglasses', 'KitchenPot', 'Basket', 'Bowl']

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def write_json(scene, category, result, cnt_dict_path):
    try:
        with open(cnt_dict_path, 'r') as json_file:
            cnt_dict = json.load(json_file)
    except:
        return
    cnt_dict[scene]['total'][result] += 1
    cnt_dict[scene]['total']['total'] += 1
    cnt_dict[scene][category][result] += 1
    cnt_dict[scene][category]['total'] += 1
    with open(cnt_dict_path, 'w') as json_file:
        json.dump(cnt_dict, json_file)

def find_scene_and_category(data_dict):
    # 找到符合条件的场景
    candidate_scenes = [scene for scene in data_dict if data_dict[scene]['total']['succ'] < args.max_num*10]
    # 如果没有符合条件的场景，返回None
    if not candidate_scenes:
        return None, None
    # 随机选择一个场景
    random.seed(datetime.datetime.now())
    selected_scene = random.choice(candidate_scenes)
    # 在所选场景中找到符合条件的类别
    candidate_categories = [category for category in data_dict[selected_scene] if category != 'total' and data_dict[selected_scene][category]['succ'] < args.max_num]
    # 如果没有符合条件的类别，返回所选场景和None
    if not candidate_categories:
        return selected_scene, None
    # 随机选择一个类别
    selected_category = random.choice(candidate_categories)
    return selected_scene, selected_category

def run_jobs(idx_process, args, transition_Q):
    random.seed(datetime.datetime.now())
    setup_seed(random.randint(1, 1000) + idx_process)
    sum_trial = args.single_trail

    for trial in range(args.start_epoch, sum_trial):
        cur_trial = sum_trial * idx_process + trial
        cur_random_seed = np.random.randint(10000000)
        try:
            with open(cnt_dict_path, 'r') as json_file:
                cnt_dict = json.load(json_file)
        except:
            continue

        selected_scene, selected_cat = find_scene_and_category(cnt_dict)
        if selected_scene is None:
            print('Full')
            break
    
        # print(cat2shape_dict)
        shape_id = cat2shape_dict[selected_cat][random.randint(0, len(cat2shape_dict[selected_cat]) - 1)]
        
        # print('start cmd')
        cmd = 'python collect_data.py --trial_id %d --shape_id %s --category %s --random_seed %d ' \
              '--density %f --damping %d --target_part_state %s --start_dist %f --maneuver_dist %f --displacement %f ' \
              '--move_steps %d --maneuver_steps %d --wait_steps %d --out_dir %s --scene %s --no_gui ' \
              % (cur_trial, shape_id, selected_cat, cur_random_seed,
                 args.density, args.damping, args.target_part_state, args.start_dist, args.maneuver_dist, args.displacement,
                 args.move_steps, args.maneuver_steps, args.wait_steps, args.out_dir, selected_scene)
        if cnt_dict[selected_scene][selected_cat]['fail'] < args.max_num * 2:
            cmd += '--save_fail '
        if cnt_dict[selected_scene][selected_cat]['invalid'] < args.max_num * 2:
            cmd += '--save_invalid '
        if np.random.uniform(0, 1) < args.con_ratio:
            cmd += '--con '
        cmd += '> /dev/null 2>&1'
        # print(cmd)
        
        if trial % 20 == 0:
            print(cnt_dict)

        ret = call(cmd, shell=True)
        print(f"ret = {ret}", f"seed = {cur_random_seed}")
        if ret == 2:
            with open('cmd.txt', 'a') as fout:
                fout.write(cmd + '\n')

        if ret == 1:
            transition_Q.put(['fail', selected_scene, selected_cat, trial])

        if ret ==2:
                transition_Q.put(['invalid', selected_scene, selected_cat, trial])

        if ret == 0:
            transition_Q.put(['succ', selected_scene, selected_cat, trial])     # dual succ

            
if __name__ == '__main__':
    # create the out directory
    out_dir = args.out_dir
    print('out_dir: ', out_dir)
    if os.path.exists(out_dir):
        response = input('Out directory "%s" already exists, continue? (y/n) ' % out_dir)
        if response != 'y' and response != 'Y':
            sys.exit()


    if not os.path.exists(out_dir):
        os.makedirs(os.path.join(out_dir, 'succ_gif'))
        os.makedirs(os.path.join(out_dir, 'fail_gif'))
        os.makedirs(os.path.join(out_dir, 'invalid_gif'))
        # os.makedirs(os.path.join(out_dir, 'tmp_succ_gif'))

        os.makedirs(os.path.join(out_dir, 'succ_files'))
        os.makedirs(os.path.join(out_dir, 'fail_files'))
        os.makedirs(os.path.join(out_dir, 'invalid_files'))
        # os.makedirs(os.path.join(out_dir, 'tmp_succ_files'))

    # Scene_list = ['table', 'wall', 'groove', 'slope']

    cat_list, shape_list, shape2cat_dict, cat2shape_dict = utils.get_shape_list(all_categories=args.category, mode=args.mode)
    scene_list = args.scene.split(',')

    Thin_categories = [cat for cat in Thin_categories if cat in cat_list]
    Pickable_categories = [cat for cat in Pickable_categories if cat in cat_list]
    if args.initialize_dict:
        cnt_dict = dict()
        combined_categories = Thin_categories + Pickable_categories
        for scene in scene_list:
            cnt_dict[scene] = dict()
            cnt_dict[scene]['total'] = {'succ': 0, 'fail': 0, 'invalid': 0, 'total': 0}
            for category in combined_categories:
                cnt_dict[scene][category] = {'succ': 0, 'fail': 0, 'invalid': 0, 'total': 0}
        with open(cnt_dict_path, 'w') as json_file:
            json.dump(cnt_dict, json_file)
    else:
        with open(cnt_dict_path, 'r') as json_file:
            cnt_dict = json.load(json_file)

    trans_q = mp.Queue()
    processes = []
    for idx_process in range(args.num_processes):
        p = mp.Process(target=run_jobs, args=(idx_process, args, trans_q))
        p.start()
        processes.append(p)

    total, max_trial = 0, 0
    cnt_dict = {'succ': 0, 'fail': 0, 'invalid': 0}
    # print(args.con_ratio)

    t0 = time.time()
    t_begin = datetime.datetime.now()
    while True:
        if not trans_q.empty():
            result, selected_scene, selected_cat, trial_id = trans_q.get()
            cnt_dict[result] += 1
            total += 1
            max_trial = max(max_trial, trial_id)
            write_json(selected_scene, selected_cat, result, cnt_dict_path)

            print(
                'Episode: {} | trial_id: {} | Valid_portion: {:.4f} | Succ_portion: {:.4f} | Running Time: {:.4f} | Total Time:'.format(
                    total, max_trial, (cnt_dict['succ'] + cnt_dict['fail']) / total, cnt_dict['succ'] / total, time.time() - t0), datetime.datetime.now() - t_begin
            )
            t0 = time.time()

            if total >= args.total_trail:
                for p in processes:
                    print(f"terminating process {p}")
                    p.terminate()
                    p.join()
                break

