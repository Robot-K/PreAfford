import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

# https://github.com/erikwijmans/Pointnet2_PyTorch
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG


class PointNet2SemSegSSG(PointNet2ClassificationSSG):
    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=1024,
                radius=0.1,
                nsample=32,
                mlp=[4, 32, 32, 64],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.2,
                nsample=32,
                mlp=[64, 64, 64, 128],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=64,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=16,
                radius=0.8,
                nsample=32,
                mlp=[256, 256, 256, 512],
                use_xyz=True,
            )
        )

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[128 + 4, 128, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 64, 256, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 128, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + 256, 256, 256]))

        self.fc_layer = nn.Sequential(
            nn.Conv1d(128, self.hparams['feat_dim'], kernel_size=1, bias=False),
            nn.BatchNorm1d(self.hparams['feat_dim']),
            nn.ReLU(True),
        )

    def forward(self, pointcloud):
        r"""
            Forward pass of the network
            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return self.fc_layer(l_features[0])


class ActionScore(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(ActionScore, self).__init__()

        self.hidden_dim = 128
        self.mlp1 = nn.Linear(input_dim, self.hidden_dim)
        self.mlp2 = nn.Linear(self.hidden_dim, output_dim)

    # feats B x F
    # output: B
    def forward(self, inputs):
        feats = torch.cat(inputs, dim=-1)
        net = F.leaky_relu(self.mlp1(feats))
        net = self.mlp2(net)
        return net


class Network(nn.Module):
    def __init__(self, feat_dim, cp_feat_dim, dir_feat_dim):
        super(Network, self).__init__()

        self.pointnet2 = PointNet2SemSegSSG({'feat_dim': feat_dim})

        self.mlp_dir = nn.Linear(3 + 3, dir_feat_dim)
        self.mlp_cp = nn.Linear(3, cp_feat_dim)     # contact point

        self.BCELoss = nn.BCEWithLogitsLoss(reduction='none')
        self.L1Loss = nn.L1Loss(reduction='none')

        self.action_score = ActionScore(feat_dim + cp_feat_dim)


    def forward(self, pcs, cp):
        twos = (torch.ones(len(cp), 1)*1).to(cp.device)
        pcs[:, 0] = torch.cat((cp, twos), dim=1) #cp最后一维设成2，现在改成1
        last_col = pcs[:, :, -1].unsqueeze(-1)
        pcs = torch.cat((pcs[:, :, :3].repeat(1, 1, 2), last_col), dim=2)
        whole_feats = self.pointnet2(pcs)
        net = whole_feats[:, :, 0]

        cp_feats = self.mlp_cp(cp)
        pred_result_logits = self.action_score([net, cp_feats])

        pred_score = torch.sigmoid(pred_result_logits)
        return pred_score

    def get_loss(self, pred_score, gt_score):
        loss = self.L1Loss(pred_score, gt_score).mean()
        return loss


    def inference_whole_pc(self, pcs, style = '0', cp_batch_size=50):
        assert pcs.shape[0] == 1
        batch_size = pcs.shape[0]
        # print('inference_whole_pc, batch_size:', batch_size)
        num_pts = pcs.shape[1]

        cp = pcs.view(batch_size * num_pts, -1)[:, :3]
        if style == '0':
            index = np.where(pcs.cpu().numpy()[0, :, 3] == 1)[0]
            # print("index = ", index)
            index = torch.from_numpy(index).to(pcs.device)  # 将索引转换为 PyTorch 张量并确保设备一致
            cp = cp[index]

        pred_scores = torch.zeros(batch_size, num_pts).to(pcs.device)
        for start_idx in range(0, len(cp), cp_batch_size):
            end_idx = min(start_idx + cp_batch_size, len(cp))
            cur_cp = cp[start_idx:end_idx, :]
            cp_feats = self.mlp_cp(cur_cp)
            # print(cp_feats.shape)
        
            cur_pcs = pcs.repeat([len(cur_cp), 1, 1])
            twos = (torch.ones(len(cur_cp), 1)*1).to(cp.device)
            cur_pcs[:, 0] = torch.cat((cur_cp, twos), dim=1) #cp最后一维设成2
            last_col = cur_pcs[:, :, -1].unsqueeze(-1)
            cur_pcs = torch.cat((cur_pcs[:, :, :3].repeat(1, 1, 2), last_col), dim=2)
            whole_feats = self.pointnet2(cur_pcs)
            net = whole_feats[:, :, 0]
            # print(net.shape)
            pred_result_logits = self.action_score([net, cp_feats])
            batch_pred_score = torch.sigmoid(pred_result_logits).reshape(batch_size, end_idx - start_idx)

            # 将批次结果存储到相应位置
            if style == '0':
                pred_scores[:, index[start_idx:end_idx]] = batch_pred_score
            else:
                pred_scores[:, start_idx:end_idx] = batch_pred_score
        return pred_scores
    
    def inference_whole_pc2(self, pcs):
        batch_size = pcs.shape[0]
        num_pts = pcs.shape[1]

        cp = pcs.view(batch_size * num_pts, -1)[:, :3]
        cp_feats = self.mlp_cp(cp)

        last_col = pcs[:, :, -1].unsqueeze(-1)
        pcs = torch.cat((pcs[:, :, :3].repeat(1, 1, 2), last_col), dim=2)
        whole_feats = self.pointnet2(pcs)
        net1 = whole_feats.permute(0, 2, 1).reshape(batch_size * num_pts, -1)
        pred_result_logits = self.action_score([net1, cp_feats])
        pred_score = torch.sigmoid(pred_result_logits).reshape(batch_size, num_pts)
        return pred_score

