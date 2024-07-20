import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

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


class ActorEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorEncoder, self).__init__()

        self.hidden_dim = 128
        self.mlp1 = nn.Linear(input_dim, self.hidden_dim)
        self.mlp2 = nn.Linear(self.hidden_dim, output_dim)
        self.mlp3 = nn.Linear(output_dim, output_dim)
        self.get_mu = nn.Linear(output_dim, output_dim)
        self.get_logvar = nn.Linear(output_dim, output_dim)

        self.BCELoss = nn.BCEWithLogitsLoss(reduction='none')
        self.L1Loss = nn.L1Loss(reduction='none')


    def forward(self, inputs):
        net = torch.cat(inputs, dim=-1)
        net = F.leaky_relu(self.mlp1(net))
        net = F.leaky_relu(self.mlp2(net))
        net = self.mlp3(net)
        mu = self.get_mu(net)
        logvar = self.get_logvar(net)
        noise = torch.Tensor(torch.randn(*mu.shape)).cuda()
        z = mu + torch.exp(logvar / 2) * noise
        return z, mu, logvar


class ActorDecoder(nn.Module):
    def __init__(self, input_dim, output_dim=6):
        super(ActorDecoder, self).__init__()

        self.hidden_dim = 128
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, output_dim)
        )

    def forward(self, inputs):
        net = torch.cat(inputs, dim=-1)
        net = self.mlp(net)
        return net


class Network(nn.Module):
    def __init__(self, feat_dim, cp_feat_dim, task_feat_dim, z_dim=128, lbd_kl=1.0, lbd_dir=1.0):
        super(Network, self).__init__()

        self.feat_dim = feat_dim
        self.z_dim = z_dim

        self.lbd_kl = lbd_kl
        self.lbd_dir = lbd_dir

        self.pointnet2 = PointNet2SemSegSSG({'feat_dim': feat_dim})

        self.mlp_task = nn.Linear(3, task_feat_dim)
        self.mlp_cp = nn.Linear(3, cp_feat_dim)     # contact point

        self.BCELoss = nn.BCEWithLogitsLoss(reduction='none')
        self.L1Loss = nn.L1Loss(reduction='none')

        self.all_encoder = ActorEncoder(input_dim=feat_dim + cp_feat_dim + task_feat_dim, output_dim=z_dim)
        self.decoder = ActorDecoder(input_dim=feat_dim + cp_feat_dim + z_dim, output_dim=3)


    def KL(self, mu, logvar):
        mu = mu.view(mu.shape[0], -1)
        logvar = logvar.view(logvar.shape[0], -1)
        # ipdb.set_trace()
        loss = 0.5 * torch.sum(mu * mu + torch.exp(logvar) - 1 - logvar, 1)
        # high star implementation
        # torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu ** 2 - 1. - z_var, 1))
        loss = torch.mean(loss)
        return loss


    # 6D-Rot loss
    # input sz bszx6
    def get_3d_loss(self, pred_3d, gt_3d):
        bias = (pred_3d - gt_3d)**2
        bias = torch.sum(bias, dim=1)**0.8
        return bias.mean()


    def forward(self, pcs, cp, task):
        twos = (torch.ones(len(cp), 1)*1).to(cp.device)
        pcs[:, 0] = torch.cat((cp, twos), dim=1) #cp最后一维设成2
        last_col = pcs[:, :, -1].unsqueeze(-1)
        pcs = torch.cat((pcs[:, :, :3].repeat(1, 1, 2), last_col), dim=2)
        whole_feats = self.pointnet2(pcs)
        net1 = whole_feats[:, :, 0]

        cp1_feats = self.mlp_cp(cp)
        task_feats = self.mlp_task(task)

        z_all, mu, logvar = self.all_encoder([net1, cp1_feats, task_feats])
        recon_task = self.decoder([net1, cp1_feats, z_all])

        return recon_task, mu, logvar


    def get_loss(self, pcs, cp1, dir1):
        batch_size = pcs.shape[0]
        recon_dir1, mu, logvar = self.forward(pcs, cp1, dir1)
        dir_loss = self.get_3d_loss(recon_dir1, dir1)
        kl_loss = self.KL(mu, logvar)
        losses = {}
        losses['kl'] = kl_loss
        losses['dir'] = dir_loss
        losses['tot'] = self.lbd_kl * kl_loss + self.lbd_dir * dir_loss

        return losses


    def actor_sample(self, pcs, cp):
        batch_size = cp.shape[0]

        twos = (torch.ones(len(cp), 1)*1).to(cp.device)
        pcs[:, 0] = torch.cat((cp, twos), dim=1) #cp最后一维设成2
        last_col = pcs[:, :, -1].unsqueeze(-1)
        pcs = torch.cat((pcs[:, :, :3].repeat(1, 1, 2), last_col), dim=2)
        whole_feats = self.pointnet2(pcs)
        net1 = whole_feats[:, :, 0]

        cp1_feats = self.mlp_cp(cp)

        z_all = torch.Tensor(torch.randn(batch_size, self.z_dim)).cuda()

        recon_dir1 = self.decoder([net1, cp1_feats, z_all])

        return recon_dir1


    def actor_sample_n(self, pcs, cp, rvs=100):
        batch_size = pcs.shape[0]

        twos = (torch.ones(len(cp), 1)*1).to(cp.device)
        pcs[:, 0] = torch.cat((cp, twos), dim=1) #cp最后一维设成2
        last_col = pcs[:, :, -1].unsqueeze(-1)
        pcs = torch.cat((pcs[:, :, :3].repeat(1, 1, 2), last_col), dim=2)
        whole_feats = self.pointnet2(pcs)
        net1 = whole_feats[:, :, 0]

        cp1_feats = self.mlp_cp(cp)

        z_all = torch.Tensor(torch.randn(batch_size, rvs, self.z_dim)).to(net1.device)

        expanded_rvs = z_all.reshape(batch_size * rvs, -1)
        expanded_net1 = net1.unsqueeze(dim=1).repeat(1, rvs, 1).reshape(batch_size * rvs, -1)
        expanded_cp1_feats = cp1_feats.unsqueeze(dim=1).repeat(1, rvs, 1).reshape(batch_size * rvs, -1)

        recon_dir1 = self.decoder([expanded_net1, expanded_cp1_feats, expanded_rvs])
        return recon_dir1 # (batch_size * rvs, 3)


    def actor_sample_n_diffCtpts(self, pcs, cp, rvs_ctpt=10, rvs=100):
        batch_size = pcs.shape[0]

        cp1_feats = self.mlp_cp(cp)        # (B * rvs_ctpt, -1)
        z_all = torch.Tensor(torch.randn(batch_size * rvs_ctpt, rvs, self.z_dim)).to(pcs.device)
        pcs = pcs.unsqueeze(dim=1).repeat(1, 1, rvs_ctpt, 1).reshape(batch_size * rvs_ctpt, -1, 4)      # (B * rvs_ctpt) * N * 3
        twos = (torch.ones(len(cp), 1)*1).to(cp.device)
        pcs[:, 0] = torch.cat((cp, twos), dim=1) #cp最后一维设成2
        last_col = pcs[:, :, -1].unsqueeze(-1)
        pcs = torch.cat((pcs[:, :, :3].repeat(1, 1, 2), last_col), dim=2)
        
        whole_feats = self.pointnet2(pcs)
        # whole_feats = self.pointnet2(pcs)
        net1 = whole_feats[:, :, 0]

        expanded_rvs = z_all.reshape(batch_size * rvs_ctpt * rvs, -1)
        expanded_net1 = net1.unsqueeze(dim=1).repeat(1, rvs, 1).reshape(batch_size * rvs_ctpt * rvs, -1)
        expanded_cp1_feats = cp1_feats.unsqueeze(dim=1).repeat(1, rvs, 1).reshape(batch_size * rvs_ctpt * rvs, -1)

        recon_dir1 = self.decoder([expanded_net1, expanded_cp1_feats, expanded_rvs])
        return recon_dir1
    
    def actor_sample_n_finetune(self, pcs, cp1, rvs_ctpt=10, rvs=100):
        batch_size = pcs.shape[0]

        cp1_feats = self.mlp_cp(cp1)        # (B * rvs_ctpt, -1)
        z_all = torch.Tensor(torch.randn(batch_size * rvs_ctpt, rvs, self.z_dim)).to(pcs.device)
        twos = (torch.ones(len(cp1), 1)*1).to(cp1.device)
        pcs[:, 0: rvs_ctpt] = torch.cat((cp1, twos), dim=1).reshape(batch_size, rvs_ctpt, 4)
        last_col = pcs[:, :, -1].unsqueeze(-1)
        pcs = torch.cat((pcs[:, :, :3].repeat(1, 1, 2), last_col), dim=2)
        whole_feats = self.pointnet2(pcs)
        net1 = whole_feats[:, :, 0: rvs_ctpt].permute(0, 2, 1).reshape(batch_size * rvs_ctpt, -1)

        expanded_rvs = z_all.reshape(batch_size * rvs_ctpt * rvs, -1)
        expanded_net1 = net1.unsqueeze(dim=1).repeat(1, rvs, 1).reshape(batch_size * rvs_ctpt * rvs, -1)
        expanded_cp1_feats = cp1_feats.unsqueeze(dim=1).repeat(1, rvs, 1).reshape(batch_size * rvs_ctpt * rvs, -1)

        recon_dir1 = self.decoder([expanded_net1, expanded_cp1_feats, expanded_rvs])

        return recon_dir1
