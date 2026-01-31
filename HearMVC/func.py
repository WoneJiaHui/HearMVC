import torch
import torch.nn.functional as F

def k_means(X, num_clusters, num_iterations):
    device = X.device
    indices = torch.randperm(X.shape[0], device=device)[:num_clusters]
    centroids = X[indices]
    centroids = centroids.to(device)
    for _ in range(num_iterations):
        distances = torch.cdist(X, centroids)
        cluster_indices = torch.argmin(distances, dim=1)
        for i in range(num_clusters):
            mask = cluster_indices == i
            if torch.any(mask):
                centroids[i] = X[mask].mean(dim=0)

    return centroids, cluster_indices

def square_euclid_distance(Z, center):
    ZZ = (Z * Z).sum(-1).reshape(-1, 1).repeat(1, center.shape[0])
    CC = (center * center).sum(-1).reshape(1, -1).repeat(Z.shape[0], 1)
    ZZ_CC = ZZ + CC
    ZC = Z @ center.T
    distance = ZZ_CC - 2 * ZC  # 这里用了平方差公式
    return distance

def high_confidence_q(threshold, qs):
    k = int(qs[0].shape[0] * threshold)  # 高置信样本数
    # k = 230  # 高置信样本数
    indices = []
    for v in range(len(qs)):
        maxq, _ = qs[v].max(dim=1)
        _, index = torch.topk(maxq, k)
        indices.append(index)

    return indices

def target_distribution(q_):
    weight = (q_ ** 2) / torch.sum(q_, 0)
    return (weight.t() / torch.sum(weight, 1)).t()