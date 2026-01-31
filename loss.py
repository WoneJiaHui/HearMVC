import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from func import *


class Loss(nn.Module):
    def __init__(self, batch_size, class_num, temperature_f, device):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.class_num = class_num
        self.temperature_f = temperature_f
        # self.temperature_l = temperature_l
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        # self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N//2):
            mask[i, N//2 + i] = 0
            mask[N//2 + i, i] = 0
        mask = mask.bool()
        return mask

    def forward_feature_InfoNCE(self, h_i, h_j, batch_size=256):
        self.batch_size = batch_size

        N = 2 * self.batch_size
        h = torch.cat((h_i, h_j), dim=0)

        sim = torch.matmul(h, h.T) / self.temperature_f
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss


    def hard_infoNCE(self, Z1, Z2, node_num, Q1, Q2, H1, H2):
        # Q是伪标签 目前并没有将低置信的权重设为1
        Q1 = torch.tensor(Q1).cuda()
        Q2 = torch.tensor(Q2).cuda()
        label_relation = Q1.view(1, -1) == Q2.view(-1, 1)
        label_relation_2view = torch.cat([torch.cat([label_relation, label_relation], dim=1),
                                          torch.cat([label_relation, label_relation], dim=1)], dim=0)
        label_relation_2view = label_relation_2view.float()

        # H是高置信索引
        H = torch.cat([H1, H2 + node_num], dim=0)
        H_mat = np.ix_(H.cpu(), H.cpu())

        Z1 = F.normalize(Z1, dim=1, p=2)
        Z2 = F.normalize(Z2, dim=1, p=2)
        S = torch.cat([torch.cat([Z1 @ Z1.T, Z1 @ Z2.T], dim=1),
                       torch.cat([Z2 @ Z1.T, Z2 @ Z2.T], dim=1)], dim=0)
        S = (S + 1) / 2
        # S_ = (2 - S.min()) / (S.max() - S.min()) * (S - S.min()) + S.min()

        # 初始化权重矩阵为全1
        pos_neg_weight = torch.ones([node_num * 2, node_num * 2]).cuda()
        # 对权重矩阵的高置信位置赋值
        temp = (label_relation_2view - S).abs()[H_mat]
        # 缩放困难权重
        # temp = (2 - temp.min()) / (temp.max() - temp.min()) * (temp - temp.min()) + temp.min()
        # temp = 2 * temp
        pos_neg_weight[H_mat] = temp.data
        pos_neg_weight = pos_neg_weight.detach()

        M = (torch.ones([node_num * 2, node_num * 2]) - torch.eye(node_num * 2)).cuda()  # mask
        pos_neg = M * torch.exp(S * pos_neg_weight)
        pos = torch.cat([torch.diag(S * pos_neg_weight, node_num), torch.diag(S * pos_neg_weight, -node_num)], dim=0)
        pos = torch.exp(pos)
        neg = (torch.sum(pos_neg, dim=1) - pos)
        infoNEC = (-torch.log(pos / (pos + neg))).sum() / (2 * node_num)
        return infoNEC

    def cluster_infoNCE(self, q_i, q_j):
        eps = 1e-8
        p_i = q_i.sum(0).view(-1)
        p_i /= p_i.sum()
        p_i = torch.clamp(p_i, min=eps)
        ne_i = (p_i * torch.log(p_i)).sum()
        p_j = q_j.sum(0).view(-1)
        p_j /= p_j.sum()
        p_j = torch.clamp(p_j, min=eps)
        ne_j = (p_j * torch.log(p_j)).sum()
        entropy = ne_i + ne_j  # 正则项，防止所有样本被分配到单一聚类

        q_i = q_i.t()
        q_j = q_j.t()
        N = 2 * self.class_num
        q = torch.cat((q_i, q_j), dim=0)
        sim = nn.CosineSimilarity(dim=2)(q.unsqueeze(1), q.unsqueeze(0)) / self.temperature_f
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)
        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_clusters = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        entropy /= N
        return loss + entropy
