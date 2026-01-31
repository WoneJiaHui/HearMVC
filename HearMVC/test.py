import scipy.io
import numpy as np

mat_path = 'data/MSRCv1.mat'
mat_data = scipy.io.loadmat(mat_path)

# 打印完整维度信息
print("原始X的完整形状：", mat_data['X'].shape)
print("原始X的维度数：", mat_data['X'].ndim)
print("原始Y的完整形状：", mat_data['Y'].shape)

# 打印X的前5个元素（辅助判断）
print("\nX前5个元素：", mat_data['X'].flatten()[:5])