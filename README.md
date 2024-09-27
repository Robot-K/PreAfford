# PreAfford: Universal Affordance-Based Pre-Grasping for Diverse Objects and Environments


![Overview](./images/teaser.png)

**The Proposed DualAfford Task.** Given different shapes and manipulation tasks (*e.g.*, pushing the keyboard in the direction indicated by the red arrow), our proposed *DualAfford* framework predicts dual collaborative visual actionable affordance and gripper orientations. The prediction for the second gripper (b) is dependent on the first (a). We can directly apply our network to real-world data.

## About the paper

PreAfford is accepted to IROS 2023.

Our team: [Kairui Ding](https://robot-k.github.io), [Boyuan Chen](), [Ruihai Wu](https://warshallrho.github.io), [Yuyang Li](https://yuyangli.com), [Zongzheng Zhang](), [Huan-ang Gao](https://c7w.tech/about) [Siqi Li](), [Guyue Zhou](), [Yixin Zhu](https://yzhu.io) and [Hao Zhao](https://sites.google.com/view/fromandto).

Arxiv Version: https://arxiv.org/abs/2404.03634.pdf

Project Page: https://air-discover.github.io/PreAfford/

## Before start
To train the models, please first go to the `data` folder and download the [PartNet_Mobility](https://sapien.ucsd.edu/downloads) and [ShapeNet](https://huggingface.co/ShapeNet) datasets for PreAfford. Environmental assets and additional object assets can be downloaded from [here](https://drive.google.com/drive/folders/1Xhc0kP63EVD7k7RrB0hijZ0nhyclbFTz?usp=sharing).

## Dependencies
This code has been tested on Ubuntu 18.04 with Cuda 10.1, Python 3.6, and PyTorch 1.7.0.

First, install SAPIEN

    pip install http://storage1.ucsd.edu/wheels/sapien-dev/sapien-0.8.0.dev0-cp36-cp36m-manylinux2014_x86_64.whl


Then, install PointNet++.

    git clone --recursive https://github.com/erikwijmans/Pointnet2_PyTorch
    cd Pointnet2_PyTorch
    # [IMPORTANT] comment these two lines of code:
    #   https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/pointnet2_ops_lib/pointnet2_ops/_ext-src/src/sampling_gpu.cu#L100-L101
    pip install -r requirements.txt
    pip install -e .

Finally, run the following to install other packages.

    # make sure you are at the repository root directory
    pip install -r requirements.txt

For visualization, please install blender v2.79 and put the executable in your environment path.

## Generate Offline Training Data
Before training the network, we need to collect a large set of interaction trials via random sampling. The data is collected in two stages: grasping and pre-grasping.

To generate offline training data for the GRASPING phase, run the following command:

    sh scripts/run_collect_random_train_sec.sh

To generate offline training data for the PRE-GRASPING phase, run the following command:

    sh scripts/run_collect_random_train_fir.sh

We mentioned in our paper that the generated pointclouds are normalized and transformed. To perform the transformation, please run the following command:

    sh scripts/run_process_pc.sh

You can modify the content of the script file to generate data for different settings. Also, please modify the `num_processes` parameter to specify the number of CPU cores to use.

## Training Pipeline for the DualAfford Framework

#### Train the Grasping Module

As mentioned in our paper, we start with training the Grasping Module M2. We first train its Critic Module (C2) and Proposal Module (P2) simultaneously (the Proposal Module is also noted as actor in our code).

To train C2 and P2, run:

    sh scripts/run_train_critic_sec.sh
    sh scripts/run_train_actor_sec.sh

Then, after pretraining C2 and P2, we can train the Affordance Module (A2) for the Second Girpper Module. Please specify the checkpoints of the pretrained C2 and P2.

    sh scripts/run_train_affordance_sec.sh

#### Train the First Gripper Module (M1)

The First Gripper Module (M1) is trained with the awareness of the trained M2. In M1, we also train its Critic Module (C1) and Proposal Module (P1) first. Note that to train C1, we should specify the pre-trained checkpoint of the Second Gripper Module (M2).

To train C1 and P1, run:

    sh scripts/run_train_critic_fir.sh
    sh scripts/run_train_actor_fir.sh

Then, after pretraining C1 and P1, we can train the Affordance Module (A1) for the Frist Girpper Module. Please specify the checkpoints of the pretrained C1 and P1.

    sh scripts/run_train_affordance_fir.sh

## Evaluation

To enable a fast evaluation process, we manually divide the evaluation to data generation, inference, and manipulation procedure for the pre-grasping and grasping phases. Please specify the dataset folder when performing the next evaluation step.

To generate initial object poses, please run

    sh scripts/run_eval_generate_dataset.sh

To infer and manipulate for the pre-grasping phase, please run

    sh scripts/run_eval_infer_fir.sh
    sh scripts/run_eval_manipulate_fir.sh

To infer and manipulate for the grasping phase, please run

    sh scripts/run_eval_infer_sec.sh
    sh scripts/run_eval_manipulate_sec.sh


## Citations

```
@article{ding2024preafford,
  title={PreAfford: Universal Affordance-Based Pre-Grasping for Diverse Objects and Environments},
  author={Ding, Kairui and Chen, Boyuan and Wu, Ruihai and Li, Yuyang and Zhang, Zongzheng and Gao, Huan-ang and Li, Siqi and Zhu, Yixin and Zhou, Guyue and Dong, Hao and others},
  journal={arXiv preprint arXiv:2404.03634},
  year={2024}
}
```
