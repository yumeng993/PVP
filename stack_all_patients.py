"""
获取所有患者的子文件名称，将T1FS，T2FS数组归一化分别对应保存起来
确保同一患者的T1、T2FS和label一一对应
"""
import os
import numpy as np

# 父文件夹路径
parent_dir = '/home/data/CNN/127/resizedT1stack'
T1_array = []
T2FS_array = []
label_array = []


def list_subfolders(parent_dir):
    """获取所有子文件夹的名称（仅名称，不包含完整路径）"""
    patient_names = []
    for item in os.listdir(parent_dir):
        item_path = os.path.join(parent_dir, item)
        if os.path.isdir(item_path):  # 判断是否为子文件夹
            patient_names.append(item)  # ✅ 只添加文件夹名称，不是完整路径
    return sorted(patient_names)  # ✅ 排序确保顺序一致


def normalize_and_convert(array):
    """归一化到 [0, 1] 并转换为 float32"""
    array_min = array.min()
    array_max = array.max()

    # 避免除以零的情况
    if array_max - array_min == 0:
        print(f" Warning: Array has no variation, using zeros.")
        return np.zeros_like(array, dtype=np.float32)

    normalized_array = (array - array_min) / (array_max - array_min)
    normalized_array = normalized_array.astype(np.float32)

    return normalized_array


def extract_array(T1direction, T2FSdirection, patient_names, T1_array, T2FS_array, label_array):
    """
    提取并对齐T1、T2FS和标签数据
    """
    print(f"\n开始处理 {len(patient_names)} 个患者的数据...\n")

    # 用于统计
    total_processed = 0
    error_count = 0

    for idx, patient_name in enumerate(patient_names):
        # 构建子文件夹路径
        T1sub_dir = os.path.join(T1direction, patient_name)
        T2FSsub_dir = os.path.join(T2FSdirection, patient_name)

        # 检查文件夹是否存在
        if not os.path.exists(T1sub_dir):
            print(f"错误: T1 文件夹不存在: {T1sub_dir}")
            error_count += 1
            continue

        if not os.path.exists(T2FSsub_dir):
            print(f" 错误: T2FS 文件夹不存在: {T2FSsub_dir}")
            error_count += 1
            continue

        # 获取所有 .npy 文件并排序（确保T1和T2FS的文件顺序一致）
        T1_files = sorted([f for f in os.listdir(T1sub_dir) if f.endswith(".npy")])
        T2FS_files = sorted([f for f in os.listdir(T2FSsub_dir) if f.endswith(".npy")])

        # 检查文件数量是否一致
        if len(T1_files) != len(T2FS_files):
            print(
                f"  患者 {patient_name} 的 T1 ({len(T1_files)} 个) 和 T2FS ({len(T2FS_files)} 个) 文件数量不一致！")
            error_count += 1
            continue

        # 检查文件名是否一致
        if T1_files != T2FS_files:
            print(f" 患者 {patient_name} 的 T1 和 T2FS 文件名不匹配！")
            print(f"   T1 文件: {T1_files[:3]}...")
            print(f"   T2FS 文件: {T2FS_files[:3]}...")
            error_count += 1
            continue

        # 逐个处理文件
        patient_file_count = 0
        for file_name in T1_files:
            try:
                # 构建完整的文件路径
                T1_file_path = os.path.join(T1sub_dir, file_name)
                T2FS_file_path = os.path.join(T2FSsub_dir, file_name)

                # 读取 T1 数组
                array_T1 = np.load(T1_file_path)
                array_T1 = normalize_and_convert(array_T1)

                # 读取 T2FS 数组
                array_T2FS = np.load(T2FS_file_path)
                array_T2FS = normalize_and_convert(array_T2FS)

                # 检查形状是否一致
                if array_T1.shape != array_T2FS.shape:
                    print(f"错误: {patient_name}/{file_name} 的 T1 和 T2FS 形状不一致！")
                    print(f"   T1 形状: {array_T1.shape}, T2FS 形状: {array_T2FS.shape}")
                    error_count += 1
                    continue

                # 获取标签（假设文件名第一个字符是标签）
                label = int(file_name[0])

                # 添加到数组
                T1_array.append(array_T1)
                T2FS_array.append(array_T2FS)
                label_array.append(label)

                patient_file_count += 1
                total_processed += 1

            except Exception as e:
                print(f"错误: 处理文件 {patient_name}/{file_name} 时出错: {e}")
                error_count += 1
                continue

        # 打印每个患者的处理进度
        print(f" [{idx + 1}/{len(patient_names)}] 患者 {patient_name}: 处理了 {patient_file_count} 个文件")

    unique_labels, counts = np.unique(label_array, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"  标签 {label}: {count} 个样本")

    # 转换为 NumPy 数组并保存
    if len(T1_array) > 0:
        T1_array = np.stack(T1_array)
        T2FS_array = np.stack(T2FS_array)
        label_array = np.array(label_array)
        # 保存数组
        output_dir = '/home/data/CNN/127/'
        os.makedirs(output_dir, exist_ok=True)

        output_file1 = os.path.join(output_dir, 'T1_data.npy')
        output_file2 = os.path.join(output_dir, 'T2FS_data.npy')
        output_file3 = os.path.join(output_dir, 'labels.npy')

        np.save(output_file1, T1_array)
        np.save(output_file2, T2FS_array)
        np.save(output_file3, label_array)
        print(f"数据已保存:")
# 主程序
if __name__ == "__main__":
    # 设置路径
    T1direction = '/home/data/CNN/127/resizedT1stack'
    T2FSdirection = '/home/data/CNN/127/resizedT2FSstack'

    # 获取患者列表
    patient_names = list_subfolders(parent_dir)

    # 提取数组
    result = extract_array(T1direction, T2FSdirection, patient_names, T1_array, T2FS_array, label_array)
