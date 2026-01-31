import scipy.io
import numpy as np

# ===================== 路径配置 =====================
original_mat_path = 'data/MSRCv1.mat'  # 你的原始mat文件路径
new_mat_path = 'data/MSRCv1.mat'  # 生成的新mat文件路径
# ====================================================

# 1. 读取原始mat文件
mat_data = scipy.io.loadmat(original_mat_path)
X_original = mat_data['X']  # shape=(6,1)，内部是6个视图的特征数组
Y_original = mat_data['Y']  # shape=(210,1)

# 2. 提取并解析5个有效视图的特征（X1-X5）
# 从(6,1)的数组中提取前5个视图的特征（第6个忽略）
view1 = X_original[0, 0].astype(np.float32)  # X1: (210, 1302)
view2 = X_original[1, 0].astype(np.float32)  # X2: (210, 48)
view3 = X_original[2, 0].astype(np.float32)  # X3: (210, 512)
view4 = X_original[3, 0].astype(np.float32)  # X4: (210, 100)
view5 = X_original[4, 0].astype(np.float32)  # X5: (210, 256)

# 3. 验证每个视图的样本数（必须为210）
assert view1.shape[0] == 210, f"X1样本数错误：{view1.shape[0]}（预期210）"
assert view2.shape[0] == 210, f"X2样本数错误：{view2.shape[0]}（预期210）"
assert view3.shape[0] == 210, f"X3样本数错误：{view3.shape[0]}（预期210）"
assert view4.shape[0] == 210, f"X4样本数错误：{view4.shape[0]}（预期210）"
assert view5.shape[0] == 210, f"X5样本数错误：{view5.shape[0]}（预期210）"

# 4. 处理标签Y：转为(210,)的int32格式（匹配原代码）
Y_original = Y_original.squeeze().astype(np.int32).reshape(210,)

# 5. 构建新的mat文件字典（包含X1-X5和Y）
new_mat = {
    '__header__': mat_data['__header__'],
    '__version__': mat_data['__version__'],
    '__globals__': mat_data['__globals__'],
    'X1': view1,
    'X2': view2,
    'X3': view3,
    'X4': view4,
    'X5': view5,
    'Y': Y_original
}

# 6. 保存新的.mat文件（兼容原代码读取）
scipy.io.savemat(new_mat_path, new_mat, do_compression=True)

# 打印验证信息（确认生成成功）
print("✅ 新.mat文件生成完成！路径：", new_mat_path)
print("📌 各字段维度验证：")
print(f"   X1: {new_mat['X1'].shape}（预期(210, 1302)）")
print(f"   X2: {new_mat['X2'].shape}（预期(210, 48)）")
print(f"   X3: {new_mat['X3'].shape}（预期(210, 512)）")
print(f"   X4: {new_mat['X4'].shape}（预期(210, 100)）")
print(f"   X5: {new_mat['X5'].shape}（预期(210, 256)）")
print(f"   Y:  {new_mat['Y'].shape}（预期(210,)）")
print(f"   X1数据类型：{new_mat['X1'].dtype}（预期float32）")
print(f"   Y数据类型：{new_mat['Y'].dtype}（预期int32）")