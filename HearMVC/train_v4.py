# v3 将聚类损失从单视图变为跨视图
# v4 infoNCE训练完后，使用困难对比 低

import torch
from network import Network
from metric import valid
from torch.utils.data import Dataset
import numpy as np
import argparse
from loss import *
from dataloader import load_data
import os
import time
from scipy.optimize import linear_sum_assignment
from func import *
import warnings

warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Dataname = 'DHA'
# Dataname = 'CCV'
# Dataname = 'NUSWIDE'
# Dataname = 'Caltech'
# Dataname = 'Caltech-2V'
# Dataname = 'Caltech-3V'
# Dataname = 'Caltech-4V'
# Dataname = 'Caltech-5V'
Dataname = 'msrcv1'
# Dataname = 'scene'


parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)  # 256
parser.add_argument("--temperature_f", default=1.0)  # 1.0
parser.add_argument("--Lambda", default=10)  # 0.001 0.01 0.1 1 10 100 1000  重构权重
parser.add_argument("--Gama", default=0.5)  # 0.001 0.01 0.1 1 10 100 1000  kmeans指导权重
parser.add_argument("--learning_rate", default=0.0003)  # 0.0003
parser.add_argument("--weight_decay", default=0.)  # 0.
parser.add_argument("--mse_epochs", default=100)  # 100
parser.add_argument("--con_epochs", default=100)  # 100
parser.add_argument("--finetune_epochs", default=150)  # 100
parser.add_argument("--feature_dim", default=512)  # 512
parser.add_argument("--high_feature_dim", default=128)  # 128
parser.add_argument("--threshold", default=0.9)  # 0.9
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.dataset == "msrcv1":
    args.con_epochs = 10  #150  50     10
    args.Lambda = 10  # 10

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True


accs = []
nmis = []
# aris = []
# purs = []
# ACC_tmp = 0


for Runs in range(1):  # 10
    print("ROUND:{}".format(Runs + 1))
    maxacc = 0
    maxnmi = 0
    maxacc2 = 0
    maxnmi2 = 0
    acc = 0
    nmi = 0

    t1 = time.time()
    setup_seed(Runs)  # if we find that the initialization of networks is sensitive, we can set a seed for stable performance.
    dataset, dims, view, data_size, class_num = load_data(args.dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        # drop_last=True,
        drop_last=False,
    )


    def Low_level_rec_train():
        tot_loss = 0.
        criterion = torch.nn.MSELoss()  # 定义MSE损失函数：MSE = 1/n * Σ(x - xr)²

        for batch_idx, (xs, _, _) in enumerate(data_loader):
            # 1. 把多视图特征放到GPU
            for v in range(view):
                xs[v] = xs[v].to(device)

            # 2. 前向传播：生成重构特征xrs
            optimizer.zero_grad()
            _, _, xrs = model(xs)

            # 3. 计算每个视图的重构损失并求和
            loss_list = []
            for v in range(view):
                # 核心：计算第v视图“原始特征xs[v]”和“重构特征xrs[v]”的MSE
                loss_list.append(criterion(xs[v], xrs[v]))
            loss = sum(loss_list)       # 多视图重构损失总和

            # 4. 反向传播：计算损失对所有模型参数的梯度
            loss.backward()

            # 5. 梯度下降：更新编码器/解码器参数（让损失变小）
            optimizer.step()
            tot_loss += loss.item()
        # print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))


    def High_level_contrastive_train():
        tot_loss = 0.
        mes = torch.nn.MSELoss()

        for batch_idx, (xs, ys, _) in enumerate(data_loader):# 遍历批量数据
# ========== 步骤1：数据预处理 ==========
            for v in range(view):
                xs[v] = xs[v].to(device)

            optimizer.zero_grad()   # 清空上一轮梯度（避免积累）

# ========== 步骤2：模型前向传播（核心） ==========
            # zs：高维对比特征 [view, batch_size, high_feature_dim]（如[5,256,128]）
            # qs：聚类预测概率 [view, batch_size, class_num]（如[5,256,10]）
            # xrs：重构特征 [view, batch_size, 各视图原始维度]（如[5,256,512]）
            zs, qs, xrs = model(xs)

            loss_list = []

            xnum = xs[0].shape[0]

            pseudo_labels = []
            cluster_centers = []
# ========== 步骤3：KMeans聚类 + 标签对齐（无监督伪标签生成） ==========
            one_hot_labels = [] #储存对齐后的KMeans为标签
            pred_labels = []

            for v in range(view):
                # 3.1 模型预测标签：对qs[v]取argmax，得到每个样本的聚类预测标签
                predicted_labels = torch.argmax(qs[v], dim=1)
                pred_labels.append(predicted_labels)
                unique_labels = torch.unique(predicted_labels)
                k = unique_labels.size(0)
                # 3.2 对高维对比特征zs[v]做KMeans聚类（func.py的k_means函数）
                # centroids：聚类中心 [class_num, high_feature_dim]；labels：KMeans聚类标签 [batch_size]
                centroids, labels = k_means(zs[v], k, 50)
                cluster_centers.append(centroids)
                similarity_matrix = torch.zeros((torch.max(predicted_labels) + 1,
                                                 torch.max(labels) + 1))  # pred是mlp输出的聚类标签，labels是h做kmeans聚类得到的聚类标签
                similarity_matrix = similarity_matrix.cuda()  # 这里只是构建一个zero矩阵，下面做匈牙利算法，对齐两个聚类分配

                for i, j in zip(predicted_labels, labels):
                    similarity_matrix[i, j] += 1

                similarity_matrix_cpu = similarity_matrix.cpu()
                row_ind, col_ind = linear_sum_assignment(-similarity_matrix_cpu)
                y_pred_mapped = torch.clone(labels)

                for i, j in zip(col_ind, row_ind):
                    y_pred_mapped[labels == i] = j

                one_hot_label = torch.zeros((labels.size(0), class_num), device=labels.device)
                one_hot_label.scatter_(1, y_pred_mapped.unsqueeze(1), 1)
                one_hot_labels.append(one_hot_label)
                pseudo_labels.append(y_pred_mapped)

# ========== 步骤4：多损失联合计算（核心） ==========
            for v in range(view):
                # 4.1 跨视图InfoNCE对比损失（所有视图两两对比）
                for w in range(view):
                    # criterion.forward_feature_InfoNCE：计算zs[v]和zs[w]的跨视图对比损失
                    tmp = criterion.forward_feature_InfoNCE(zs[v], zs[w], batch_size=xnum)
                    loss_list.append(tmp)

                # kmeans指导损失
                kmeans_guide_loss = criterion.cluster_infoNCE(qs[v], one_hot_labels[v].detach())
                loss_list.append(args.Gama * kmeans_guide_loss)
                # 重构正则损失
                loss_list.append(args.Lambda * mes(xs[v], xrs[v]))
            # 总损失求和
            loss = sum(loss_list)
            # print('Epoch {} loss = {}'.format(epoch, loss))
# ========== 步骤5：反向传播 + 参数更新 ==========
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()



    def finetune_train():
        tot_loss = 0.
        mes = torch.nn.MSELoss()

        for batch_idx, (xs, ys, _) in enumerate(data_loader):
            for v in range(view):
                xs[v] = xs[v].to(device)

            optimizer.zero_grad()
            zs, qs, xrs = model(xs)
            loss_list = []

            xnum = xs[0].shape[0]

            pseudo_labels = []
            cluster_centers = []
            one_hot_labels = []
            pred_labels = []

            for v in range(view):
                predicted_labels = torch.argmax(qs[v], dim=1)
                pred_labels.append(predicted_labels)
                unique_labels = torch.unique(predicted_labels)
                k = unique_labels.size(0)
                centroids, labels = k_means(zs[v], k, 50)
                cluster_centers.append(centroids)
                similarity_matrix = torch.zeros((torch.max(predicted_labels) + 1,
                                                 torch.max(labels) + 1))  # pred是mlp输出的聚类标签，labels是h做kmeans聚类得到的聚类标签
                similarity_matrix = similarity_matrix.cuda()  # 这里只是构建一个zero矩阵，下面做匈牙利算法，对齐两个聚类分配

                for i, j in zip(predicted_labels, labels):
                    similarity_matrix[i, j] += 1

                similarity_matrix_cpu = similarity_matrix.cpu()
                row_ind, col_ind = linear_sum_assignment(-similarity_matrix_cpu)
                y_pred_mapped = torch.clone(labels)

                for i, j in zip(col_ind, row_ind):
                    y_pred_mapped[labels == i] = j

                one_hot_label = torch.zeros((labels.size(0), class_num), device=labels.device)
                one_hot_label.scatter_(1, y_pred_mapped.unsqueeze(1), 1)
                one_hot_labels.append(one_hot_label)
                pseudo_labels.append(y_pred_mapped)

            threshold = args.threshold
            Hs = high_confidence_q(threshold, qs)


            for v in range(view):
                for w in range(view):
                    tmp = criterion.hard_infoNCE(zs[v], zs[w], xnum, pred_labels[v], pred_labels[w], Hs[v], Hs[w])
                    loss_list.append(tmp)

                # kmeans指导损失
                kmeans_guide_loss = criterion.cluster_infoNCE(qs[v], one_hot_labels[v])
                loss_list.append(args.Gama * kmeans_guide_loss)

                loss_list.append(args.Lambda * mes(xs[v], xrs[v]))
            loss = sum(loss_list)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()

    if not os.path.exists('./models'):
        os.makedirs('./models')

    model = Network(view, dims, args.feature_dim, args.high_feature_dim, class_num, device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = Loss(args.batch_size, class_num, args.temperature_f, device).to(device)

    print("Reconstruction Regularization...")
    epoch = 0
    while epoch < args.mse_epochs:
        epoch += 1
        Low_level_rec_train()
        if epoch == args.mse_epochs:
            acc, nmi, ari, pur = valid(epoch, model, device, dataset, view, data_size, class_num)

    print("Multi-view Contrastive Learning with Reconstruction Regularization...")
    epoch = 0
    while epoch < args.con_epochs:
        epoch += 1
        High_level_contrastive_train()

        if epoch % 50 == 0:
            acc, nmi, ari, pur = valid(epoch, model, device, dataset, view, data_size, class_num)
            if acc > maxacc:
                maxacc = acc
                maxnmi = nmi

        pg = [p for p in model.parameters() if p.requires_grad]
        #  this code matters, to re-initialize the optimizers
        optimizer = torch.optim.Adam(pg, lr=args.learning_rate, weight_decay=args.weight_decay)

    print("Multi-view Hard Sample Aware Contrastive Learning with Reconstruction Regularization...")
    epoch = 0
    while epoch < args.finetune_epochs:
        epoch += 1
        finetune_train()

        if epoch % 50 == 0:
            acc, nmi, ari, pur = valid(epoch, model, device, dataset, view, data_size, class_num)
            if acc > maxacc2:
                maxacc2 = acc
                maxnmi2 = nmi

        pg = [p for p in model.parameters() if p.requires_grad]
        #  this code matters, to re-initialize the optimizers
        optimizer = torch.optim.Adam(pg, lr=args.learning_rate, weight_decay=args.weight_decay)

    accs.append(maxacc)
    nmis.append(maxnmi)

    # if acc > ACC_tmp:
    #     ACC_tmp = acc
    #     state = model.state_dict()
    #     torch.save(state, './models/' + args.dataset + '.pth')

    t2 = time.time()
    print("Time cost: " + str(t2 - t1))
    print('End......')


print(np.mean(accs) / 0.01, np.std(accs) / 0.01)
print(np.mean(nmis) / 0.01, np.std(nmis) / 0.01)
# print(aris, np.mean(aris)/0.01, np.std(aris)/0.01)
# print(purs, np.mean(purs)/0.01, np.std(purs)/0.01)
