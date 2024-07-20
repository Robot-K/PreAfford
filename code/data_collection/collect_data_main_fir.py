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
parser.add_argument('--move_steps', type=int, default=3500)
parser.add_argument('--maneuver_steps', type=int, default=1500)
parser.add_argument('--wait_steps', type=int, default=1000)
parser.add_argument('--given_direction_ratio', type=float, default=0.5)

parser.add_argument('--single_trail', type=int, default=10000)
parser.add_argument('--total_trail', type=int, default=50000)
parser.add_argument('--num_processes', type=int, default=1)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--save_interval', type=int, default=5)
parser.add_argument('--no_gui', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--use_edge', action='store_true', default=False)
parser.add_argument('--initialize_dict', action='store_true', default=False)
parser.add_argument('--not_check_dual', action='store_true', default=False)

args = parser.parse_args()
cat_cnt_dict_path = args.out_dir+'/cat_cnt_dict.json'
scene_cnt_dict_path = args.out_dir+'/scene_cnt_dict.json'

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def select_key_from_low_half(d, res=None):
    # 如果提供了 res，则只考虑 res 中的键
    if res is not None:
        d = {k: v for k, v in d.items() if k in res}
    # 计算中位数
    median_value = sorted(d.values())[len(d) // 2]
    # 找出所有值小于或等于中位数的键
    keys_in_low_half = [k for k, v in d.items() if v <= median_value]
    # 随机选择一个键并返回
    return random.choice(keys_in_low_half) if keys_in_low_half else None


def run_jobs(idx_process, args, transition_Q):
    random.seed(datetime.datetime.now())
    setup_seed(random.randint(1, 1000) + idx_process)
    sum_trial = args.single_trail

    for trial in range(args.start_epoch, sum_trial):
        cur_trial = sum_trial * idx_process + trial
        cur_random_seed = np.random.randint(10000000)
        with open(cat_cnt_dict_path, 'r') as json_file:
            cat_cnt_dict = json.load(json_file)
        with open(scene_cnt_dict_path, 'r') as json_file:
            scene_cnt_dict = json.load(json_file)

        # load object
        if  random.uniform(0, 1) < 0.5:
            selected_cat = select_key_from_low_half(cat_cnt_dict, Thin_categories)
        else:
            selected_cat = select_key_from_low_half(cat_cnt_dict, Pickable_categories)
        
        scene = select_key_from_low_half(scene_cnt_dict, args.scene.split(','))
        shape_id = cat2shape_dict[selected_cat][random.randint(0, len(cat2shape_dict[selected_cat]) - 1)]
        # shape_id = 13004
        # selected_cat = "Keyboard"
        # print('shape_id: ', shape_id, selected_cat, cur_trial)
        # scene = scene_list[random.randint(0, len(scene_list) - 1)]
        if_given_direction = random.uniform(0, 1) < args.given_direction_ratio

        if if_given_direction:
            cmd = 'python collect_data_fir.py --trial_id %d --shape_id %s --category %s --random_seed %d ' \
                '--density %f --damping %d --target_part_state %s --start_dist %f --maneuver_dist %f ' \
                '--move_steps %d --maneuver_steps %d --wait_steps %d --out_dir %s --scene %s --if_given_direction --no_gui ' \
                % (cur_trial, shape_id, selected_cat, cur_random_seed,
                    args.density, args.damping, args.target_part_state, args.start_dist, args.maneuver_dist,
                    args.move_steps, args.maneuver_steps, args.wait_steps, args.out_dir, scene)
            if trial % args.save_interval == 0:
                cmd += '--save_data '
            cmd += '> /dev/null 2>&1'
        else:
            cmd = 'python collect_data_fir.py --trial_id %d --shape_id %s --category %s --random_seed %d ' \
                '--density %f --damping %d --target_part_state %s --start_dist %f --maneuver_dist %f ' \
                '--move_steps %d --maneuver_steps %d --wait_steps %d --out_dir %s --scene %s --no_gui ' \
                % (cur_trial, shape_id, selected_cat, cur_random_seed,
                    args.density, args.damping, args.target_part_state, args.start_dist, args.maneuver_dist,
                    args.move_steps, args.maneuver_steps, args.wait_steps, args.out_dir, scene)
            if trial % args.save_interval == 0:
                cmd += '--save_data '
            cmd += '> /dev/null 2>&1'
        
        # print(cmd)
        if trial % 100 == 0:
            print(cat_cnt_dict)
            print(scene_cnt_dict)

        ret = call(cmd, shell=True)
        print(cmd)
        print(f"ret = {ret}", f"seed = {cur_random_seed}")
        # if ret == 139:
        with open('cmd.txt', 'a') as fout:
            fout.write(cmd + '\n')

        if ret == 1 or 2:
            transition_Q.put(['fail', trial])

        if ret != 0 and ret != 1 and ret != 2:
            transition_Q.put(['invalid', trial])

        if ret == 0:
            # check dual
            # ret0, ret1 = 1, 1
            # if not args.not_check_dual:
            #     cmd = 'python collect_data_checkDual.py --trial_id %d --random_seed %d --gripper_id %d ' \
            #           '--density %f --damping %d --target_part_state %s --start_dist %f --final_dist %f ' \
            #           '--move_steps %d --wait_steps %d --out_dir %s --no_gui ' \
            #           % (cur_trial, cur_random_seed, 0,
            #              args.density, args.damping, args.target_part_state, args.start_dist, args.final_dist,
            #              args.move_steps, args.wait_steps, args.out_dir)
            #     ret0 = call(cmd, shell=True)

            #     cmd = 'python collect_data_checkDual.py --trial_id %d --random_seed %d --gripper_id %d ' \
            #           '--density %f --damping %d --target_part_state %s --start_dist %f --final_dist %f ' \
            #           '--move_steps %d --wait_steps %d --out_dir %s --no_gui ' \
            #           % (cur_trial, cur_random_seed, 1,
            #              args.density, args.damping, args.target_part_state, args.start_dist, args.final_dist,
            #              args.move_steps, args.wait_steps, args.out_dir)
            #     ret1 = call(cmd, shell=True)

            # if ret0 == 0 or ret1 == 0:
            #     transition_Q.put(['fail', trial])     # single succ
            #     print('lala\n')
            #     copy_file(cur_trial, selected_cat, shape_id, succ=False, out_dir=args.out_dir)
            # else:
            transition_Q.put(['succ', trial])     # dual succ
            with open(cat_cnt_dict_path, 'r') as json_file:
                cat_cnt_dict = json.load(json_file)
            with open(scene_cnt_dict_path, 'r') as json_file:
                scene_cnt_dict = json.load(json_file)
            cat_cnt_dict[selected_cat] += 1
            scene_cnt_dict[scene] += 1
            with open(cat_cnt_dict_path, 'w') as json_file:
                json.dump(cat_cnt_dict, json_file)
            with open(scene_cnt_dict_path, 'w') as json_file:
                json.dump(scene_cnt_dict, json_file)
            # copy_file(cur_trial, selected_cat, shape_id, succ=True, out_dir=args.out_dir)
            


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

    Thin_categories = ['Switch', 'Laptop', 'Remote', 'Scissors', 'Window', 'Keyboard2', 'Pen', 'USB', 'Bowl_lie', 'Cap', 'Phone','Tablet']
    Pickable_categories = ['Box', 'Bucket', 'Display', 'Eyeglasses', 'Faucet', 'Kettle', 'KitchenPot', 'Pliers', 'Basket', 'Bowl']
    # Scene_list = ['table', 'wall', 'groove', 'slope']

    cat_list, shape_list, shape2cat_dict, cat2shape_dict = utils.get_shape_list(all_categories=args.category, mode=args.mode)
    scene_list = args.scene.split(',')

    Thin_categories = [cat for cat in Thin_categories if cat in cat_list]
    Pickable_categories = [cat for cat in Pickable_categories if cat in cat_list]
    if args.initialize_dict:
        cat_cnt_dict = dict()
        scene_cnt_dict = dict()
        combined_categories = Thin_categories + Pickable_categories
        for obj in combined_categories:
            cat_cnt_dict[obj] = 0
        for scene in scene_list:
            scene_cnt_dict[scene] = 0
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
        p = mp.Process(target=run_jobs, args=(idx_process, args, trans_q))
        p.start()
        processes.append(p)


    total, max_trial = 0, 0
    cnt_dict = {'succ': 0, 'fail': 0, 'invalid': 0}


    t0 = time.time()
    t_begin = datetime.datetime.now()
    while True:
        if not trans_q.empty():
            result, trial_id = trans_q.get()
            cnt_dict[result] += 1
            total += 1
            max_trial = max(max_trial, trial_id)

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

