import os
import h5py
import time
import sys
import shutil
import random
from time import strftime
from argparse import ArgumentParser
import numpy as np
import torch
import torch.utils.data
import subprocess
import utils
from tqdm import tqdm
import json
from pointnet2_ops.pointnet2_utils import furthest_point_sample
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'blender_utils'))


def get_gpu_info():
    try:
        result = subprocess.run(['gpustat'], stdout=subprocess.PIPE)
        output = result.stdout.decode('utf-8')
        return output
    except Exception as e:
        return f"Error occurred: {str(e)}"

def train(conf, data_list):
    affordance, actor, critic = None, None, None
    # load aff2 + actor2 + critic2
    aff_def = utils.get_model_module(conf.aff_version)
    affordance = aff_def.Network(conf.feat_dim, conf.cp_feat_dim, conf.dir_feat_dim)
    affordance.load_state_dict(torch.load(os.path.join(conf.aff_path, 'ckpts', '%s-network.pth' % conf.aff_eval_epoch)))
    affordance.to(conf.device)

    actor_def = utils.get_model_module(conf.actor_version)
    actor = actor_def.Network(conf.feat_dim, conf.cp_feat_dim, conf.dir_feat_dim, z_dim=conf.z_dim)
    actor.load_state_dict(torch.load(os.path.join(conf.actor_path, 'ckpts', '%s-network.pth' % conf.actor_eval_epoch)))
    actor.to(conf.device)

    critic_def = utils.get_model_module(conf.critic_version)
    critic = critic_def.Network(conf.feat_dim, conf.cp_feat_dim, conf.dir_feat_dim)
    critic.load_state_dict(torch.load(os.path.join(conf.critic_path, 'ckpts', '%s-network.pth' % conf.critic_eval_epoch)))
    critic.to(conf.device)

    # load dataset
    for i in range(len(data_list)):
        cur_dir = data_list[i]
        if not os.path.exists(cur_dir):
            print(cur_dir, 'not found')
            continue
        for root, dirs, files in os.walk(cur_dir):
            fs = []
            for file in sorted(files):
                if 'json' not in file:
                    continue
                fs.append(file)

            for file in tqdm(fs):
                result_idx = int(file.split('.')[0][7:])
                # result_idx = int(file.split('_')[1])
                # if result_idx != 1168:
                #     continue
                with open(os.path.join(cur_dir, 'result_%d.json' % (result_idx)), 'r') as fin:
                    try:
                        result_data = json.load(fin)
                    except Exception:
                        continue

                if result_data['category'] not in conf.categories.split(','):
                    continue

                try:
                    with h5py.File(os.path.join(cur_dir, 'cambase', '_XYZA_boxed_%d.h5' % (result_idx)), 'r') as fin:
                        pc1 = fin['xyza'][:].astype(np.float32)
                        pc1 = torch.from_numpy(pc1).unsqueeze(0)
                    with h5py.File(os.path.join(cur_dir, 'cambase', 'final_XYZA_boxed_%d.h5' % (result_idx)), 'r') as fin:
                        pc2 = fin['xyza'][:].astype(np.float32)
                        pc2 = torch.from_numpy(pc2).unsqueeze(0)
                except Exception:
                    print('fail to load pcs')
                    continue
                
                average_scores = []
                aff_score_list = []
                critic_score_list = []
                for pc_id, pcs in enumerate([pc1, pc2]):
                    with torch.no_grad():
                        pcs = pcs.to(conf.device)
                        # print(conf)
                        position1s, dir1s, aff_scores, critic_scores = utils.inference(affordance, actor, critic, pcs, conf, draw_aff_map=False, draw_proposal_map=False, draw_critic_map=False, out_dir=os.path.join(cur_dir, '../maps_sec_17_5'), file_id=result_idx, prefix = 'before' if pc_id == 0 else 'after', if_pre=conf.if_pre)
                        aff_score_list.append(aff_scores.tolist())
                        critic_score_list.append(critic_scores.tolist())
                        average_scores.append(np.mean(critic_scores).item())

                result_data['score_before'] = average_scores[0]
                result_data['score_after'] = average_scores[1]
                result_data['aff_score_before'] = aff_score_list[0]
                result_data['aff_score_after'] = aff_score_list[1]
                result_data['critic_score_before'] = critic_score_list[0]
                result_data['critic_score_after'] = critic_score_list[1]
                with open(os.path.join(cur_dir, 'result_%d.json' % result_idx), 'w') as fin:
                    json.dump(result_data, fin, indent=4)
                

if __name__ == '__main__':
    ### get parameters
    parser = ArgumentParser()

    # main parameters (required)
    parser.add_argument('--exp_suffix', type=str, help='exp suffix')
    parser.add_argument('--cat2freq', type=str, default=None)
    parser.add_argument('--val_cat2freq', type=str, default=None)
    parser.add_argument('--categories', type=str, default='Switch,Laptop,Scissors,Keyboard2,Pen,Cap')
    parser.add_argument('--offline_data_dir', type=str, help='data directory')
    parser.add_argument('--offline_data_dir2', type=str, default='xxx', help='data directory')
    parser.add_argument('--offline_data_dir3', type=str, default='xxx', help='data directory')
    parser.add_argument('--offline_data_dir4', type=str, default='xxx', help='data directory')
    parser.add_argument('--offline_data_dir5', type=str, default='xxx', help='data directory')
    parser.add_argument('--offline_data_dir6', type=str, default='xxx', help='data directory')
    parser.add_argument('--use_boxed_pc', action='store_true', default=False)

    # main parameters (optional)
    parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:x for using cuda on GPU number x')
    parser.add_argument('--seed', type=int, default=-1, help='random seed (for reproducibility) [specify -1 means to generate a random one]')
    parser.add_argument('--log_dir', type=str, default='../logs/score', help='exp logs directory')
    parser.add_argument('--overwrite', action='store_true', default=False, help='overwrite if exp_dir exists [default: False]')
    parser.add_argument('--resume', action='store_true', default=False, help='resume if exp_dir exists [default: False]')

    # network settings
    parser.add_argument('--img_size', type=int, default=448)
    parser.add_argument('--num_point_per_shape', type=int, default=8192)
    parser.add_argument('--feat_dim', type=int, default=128)
    parser.add_argument('--task_feat_dim', type=int, default=32)
    parser.add_argument('--cp_feat_dim', type=int, default=32)
    parser.add_argument('--dir_feat_dim', type=int, default=32)
    parser.add_argument('--no_true_false_equal', action='store_true', default=False, help='if make the true/false data loaded equally [default: False]')
    parser.add_argument('--coordinate_system', type=str, default='world')
    parser.add_argument('--loss_type', type=str, default='crossEntropy')

    parser.add_argument('--batch_size', type=int, default=1)

    parser.add_argument('--aff_version', type=str, default=None)
    parser.add_argument('--aff_path', type=str, default=None)
    parser.add_argument('--aff_eval_epoch', type=str, default=None)
    parser.add_argument('--actor_version', type=str, default=None)
    parser.add_argument('--actor_path', type=str, default=None)
    parser.add_argument('--actor_eval_epoch', type=str, default=None)
    parser.add_argument('--critic_version', type=str, default=None)
    parser.add_argument('--critic_path', type=str, default=None)
    parser.add_argument('--critic_eval_epoch', type=str, default=None)
    parser.add_argument('--rvs_proposal', type=int, default=100)
    parser.add_argument('--z_dim', type=int, default=32)
    parser.add_argument('--aff_topk', type=float, default=0.002)
    parser.add_argument('--num_ctpt', type=int, default=10)
    parser.add_argument('--critic_topk1', type=float, default=0.01)
    parser.add_argument('--total', action='store_true', default=False, help='if use double name [default: False]')
    parser.add_argument('--if_pre', action='store_true', default=False, help='if use double name [default: False]')


    # parse args
    conf = parser.parse_args()

    ### prepare before training
    # make exp_name

    # control randomness
    if conf.seed < 0:
        conf.seed = random.randint(1, 10000)
    random.seed(conf.seed)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)


    # set training device
    device = torch.device(conf.device)
    conf.device = device

    train_data_list = []
    offline_data_dir_list = [conf.offline_data_dir, conf.offline_data_dir2, conf.offline_data_dir3,
                             conf.offline_data_dir4, conf.offline_data_dir5, conf.offline_data_dir6]
    for data_dir in offline_data_dir_list:
        train_data_list.extend(utils.append_data_list(data_dir, total_dir=conf.total))
    train(conf, train_data_list)

