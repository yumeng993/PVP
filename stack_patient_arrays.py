import os
import numpy as np
"""
将患者的各序列子文件夹中的多个 .npy 文件沿新轴叠加成一个 3D 数组，并将结果保存到新的文件夹中。
"""

def stack_npy_files_to_new_parent_folder(root_folder, new_parent_folder):
    # 确保新的父文件夹存在，不存在则创建
    os.makedirs(new_parent_folder, exist_ok=True)

    # 遍历根文件夹下的所有子文件夹
    for subdir, dirs, files in os.walk(root_folder):
        npy_files = [f for f in files if f.endswith('.npy')]  # 获取所有 .npy 文件
        if npy_files:
            arrays = []

            # 对文件名进行排序，确保按顺序加载
            npy_files.sort()

            for npy_file in npy_files:
                file_path = os.path.join(subdir, npy_file)
                array = np.load(file_path)  # 读取 .npy 文件
                arrays.append(array)  # 将数组添加到列表中

            # 将所有数组沿新轴叠加，形成 3D 数组
            stacked_array = np.stack(arrays, axis=0)

            # 使用第一个 .npy 文件的名称首字母作为前缀命名
            first_file = npy_files[0]
            prefix = first_file[0]  # 提取文件名的首字母

            # 获取当前子文件夹的名称
            original_subfolder_name = os.path.basename(subdir)

            # 在新的父文件夹中创建与原子文件夹同名的新子文件夹
            new_subfolder = os.path.join(new_parent_folder, original_subfolder_name)
            os.makedirs(new_subfolder, exist_ok=True)  # 创建新文件夹，如果已存在则忽略

            # 将3D数组保存为stacked文件，文件名为 "prefix_stacked.npy"
            output_filename = f"{prefix}_stacked.npy"
            output_path = os.path.join(new_subfolder, output_filename)
            np.save(output_path, stacked_array)
            print(f"Saved stacked array as {output_filename} in {new_subfolder}")


# 使用方法
#root_folder = '/home/data/CNN/127/resizedT2FS_numpy'  # 修改为原始父文件夹的路径
#new_parent_folder = '/home/data/CNN/127/resizedT2FSstack'  # 修改为新的父文件夹的路径
root_folder = '/home/data/CNN/127/resizedT1_numpy'  # 修改为原始父文件夹的路径
new_parent_folder = '/home/data/CNN/127/resizedT1stack'  # 修改为新的父文件夹的路径
stack_npy_files_to_new_parent_folder(root_folder, new_parent_folder)
