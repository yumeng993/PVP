'''
用于随机划分训练集、验证集和测试集的实用函数
以及数据归一化和创建对比损失的样本对
'''
import os
import numpy as np
import torch
def stratified_train_val_test_split_random(T1, T2FS, labels,
                                           val_size0=10, val_size1=10,
                                           test_size0=5, test_size1=5,
                                           random_state=42,
                                           save_dir=None):
    if isinstance(labels, torch.Tensor):
        labels_cpu = labels.cpu()
        labels_np = labels_cpu.numpy()
    else:
        labels_np = np.array(labels)
        labels_cpu = torch.tensor(labels_np)

    # 获取类别0和类别1的索引
    indices_0 = np.where(labels_np == 0)[0]
    indices_1 = np.where(labels_np == 1)[0]

    print(f"\n{'=' * 50}")
    print(f"总样本数: {len(labels_np)}")
    print(f"标签为0的样本数: {len(indices_0)}")
    print(f"标签为1的样本数: {len(indices_1)}")

    # 设置随机种子
    np.random.seed(random_state)

    # 随机选择验证集
    val_indices_0 = np.random.choice(indices_0, size=val_size0, replace=False)
    val_indices_1 = np.random.choice(indices_1, size=val_size1, replace=False)

    # 从剩余数据中选择测试集
    remaining_indices_0 = np.setdiff1d(indices_0, val_indices_0)
    remaining_indices_1 = np.setdiff1d(indices_1, val_indices_1)

    test_indices_0 = np.random.choice(remaining_indices_0, size=test_size0, replace=False)
    test_indices_1 = np.random.choice(remaining_indices_1, size=test_size1, replace=False)

    # 合并索引
    val_indices = np.concatenate([val_indices_0, val_indices_1])
    test_indices = np.concatenate([test_indices_0, test_indices_1])
    all_val_test_indices = np.concatenate([val_indices, test_indices])
    train_indices = np.setdiff1d(np.arange(len(labels_np)), all_val_test_indices)

    # 打印数据集划分信息（使用 NumPy 数组进行计算）
    print(f"\n数据集划分:")
    print(
        f"  训练集: {len(train_indices):3d} 个样本 (标签0: {np.sum(labels_np[train_indices] == 0):3d}, 标签1: {np.sum(labels_np[train_indices] == 1):3d})")
    print(f"  验证集: {len(val_indices):3d} 个样本 (标签0: {val_size0:3d}, 标签1: {val_size1:3d})")
    print(f"  测试集: {len(test_indices):3d} 个样本 (标签0: {test_size0:3d}, 标签1: {test_size1:3d})")
    print(f"{'=' * 50}\n")

    # 划分数据（返回张量）
    train_data1 = T1[train_indices]
    train_data2 = T2FS[train_indices]
    train_labels = labels_cpu[train_indices]

    val_data1 = T1[val_indices]
    val_data2 = T2FS[val_indices]
    val_labels = labels_cpu[val_indices]

    test_data1 = T1[test_indices]
    test_data2 = T2FS[test_indices]
    test_labels = labels_cpu[test_indices]

    # 保存数据（如果指定了保存目录）
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        # 保存为 NumPy 数组
        np.save(os.path.join(save_dir, 'train_data1.npy'), train_data1.cpu().numpy())
        np.save(os.path.join(save_dir, 'train_data2.npy'), train_data2.cpu().numpy())
        np.save(os.path.join(save_dir, 'train_labels.npy'), train_labels.cpu().numpy())

        np.save(os.path.join(save_dir, 'val_data1.npy'), val_data1.cpu().numpy())
        np.save(os.path.join(save_dir, 'val_data2.npy'), val_data2.cpu().numpy())
        np.save(os.path.join(save_dir, 'val_labels.npy'), val_labels.cpu().numpy())

        np.save(os.path.join(save_dir, 'test_data1.npy'), test_data1.cpu().numpy())
        np.save(os.path.join(save_dir, 'test_data2.npy'), test_data2.cpu().numpy())
        np.save(os.path.join(save_dir, 'test_labels.npy'), test_labels.cpu().numpy())

        print(f"✅ 数据已保存到: {save_dir}\n")

    # 清理 GPU 内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (train_data1, train_data2, train_labels,
            val_data1, val_data2, val_labels,
            test_data1, test_data2, test_labels)


def normalize_patient_data(data):
    """
    对每位患者的数据进行归一化，将其值缩放到 [0, 1]
    """
    normalized_data = []
    for patient_data in data:
        # 使用 [0, 1] 归一化
        min_val = patient_data.min()
        max_val = patient_data.max()
        normalized = (patient_data - min_val) / (max_val - min_val + 1e-8)  # 避免除以零
        normalized_data.append(normalized)
    return np.array(normalized_data)


def create_sample_pairs_by_labels(labels):
    """
    根据标签创建正样本对和负样本对
    """
    pos_pairs = []
    neg_pairs = []
    label_to_indices = {}

    for idx, label in enumerate(labels):
        label_value = label.item() if isinstance(label, torch.Tensor) else label
        if label_value not in label_to_indices:
            label_to_indices[label_value] = []
        label_to_indices[label_value].append(idx)

    # 创建正样本对（相同标签）
    for indices in label_to_indices.values():
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                pos_pairs.append((indices[i], indices[j]))

    # 创建负样本对（不同标签）
    if isinstance(labels, torch.Tensor):
        labels_list = labels.cpu().numpy()
    else:
        labels_list = labels

    for i in range(len(labels_list)):
        for j in range(i + 1, len(labels_list)):
            if labels_list[i] != labels_list[j]:
                neg_pairs.append((i, j))

    return pos_pairs, neg_pairs