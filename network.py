import torch
import torch.nn as nn
from torch.nn.functional import normalize


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)


class Network(nn.Module):
    def __init__(self, view, input_size, feature_dim, high_feature_dim, class_num, device):
        super(Network, self).__init__()
        self.view = view
        self.encoders = []
        self.decoders = []
        self.feature_contrastive_modules = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim))
            self.decoders.append(Decoder(input_size[v], feature_dim))
            self.feature_contrastive_modules.append(
                nn.Sequential(
                    nn.Linear(feature_dim, high_feature_dim),
                    #
                    # nn.Linear(feature_dim, feature_dim),
                    # nn.ReLU(),
                    # nn.Linear(feature_dim, high_feature_dim),
                )
            )
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.feature_contrastive_modules = nn.ModuleList(self.feature_contrastive_modules)

        self.label_mlp = nn.Sequential(
            # nn.Linear(feature_dim, high_feature_dim),
            # nn.ReLU(),
            nn.Linear(high_feature_dim, class_num),
            nn.Softmax(dim=1)
        )

    def forward(self, xs):
        qs = []     # 聚类预测概率
        xrs = []    # 各视图的重构特征
        zs = []     # 对比特征
        for v in range(self.view):      # 遍历每个视图
            x = xs[v]       # 取第v个视图的原始特征（输入）
            h = self.encoders[v](x)     # encoder是特征提取器（比如MLP），h是低维编码特征
            z = normalize(self.feature_contrastive_modules[v](h), dim=1)
            q = self.label_mlp(z)
            # h = normalize(z, dim=1)
            xr = self.decoders[v](h)    # decoder是解码器（与encoder对称的MLP），xr是重构特征
            # 收集结果
            zs.append(z)
            xrs.append(xr)
            qs.append(q)
        return zs, qs, xrs

    def pred(self, z):
        q = self.label_mlp(z)
        return q
