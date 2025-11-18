#该脚本用于将DICOM图像调整为指定大小，并根据Excel文件中的标签信息对图像进行分类保存为numpy数组。
import pandas as pd
import os
import cv2
import pydicom
import numpy as np
#定义一个列表放患者名称
patient_names = []
series = []
dicom_paths = []
# 读取Excel文件
file_path = '/home/data/CNN/127/127.xlsx'
#原始图像路径,下一级菜单就是患者
directory_path = '/home/data/CNN/127/T2FS'
#directory_path = '/home/data/CNN/127/T2FS'
df = pd.read_excel(file_path)
def find_value_in_excel(df, value):
    for index, row in df.iterrows():
        if value in row.values:
            # 获取该行特定列的数值
            label = row[14]
            if label > 0 :
                label = 1
            else:
                label = 0
            return label
    return print(-1)  # 如果没有找到值，则返回 -1
def subfolder(path):
    # 获取目录中的文件夹列表
    folders = os.listdir(path)
    for folder in folders:
        patient_names.append(folder)
subfolder(directory_path)
def resize_dicom(input_dicom_folder, resized_subfolder_path, new_width, new_height,numpy_subfolder_path,label):
    # 读取原始 DICOM 文件
    files = os.listdir(input_dicom_folder)
    #dicom_files = [f for f in files if f.endswith('.dcm')]  # 筛选出以'.dcm'结尾的文件
    dicom_files = files
    for dicom_file in dicom_files:
        file_path = os.path.join(input_dicom_folder, dicom_file)
        # 读取dicom图像
        dicom_data = pydicom.dcmread(file_path)
        # 提取dicom图像数组
        pixel_array = dicom_data.pixel_array
        # 获取数组的最大值与最小值
        max_value = np.max(pixel_array)
        min_value = np.min(pixel_array)
        # 归一化到 0 到 1 的范围
        normalized_image = pixel_array.astype(np.float32)
        normalized_image = (normalized_image - normalized_image.min()) / (normalized_image.max() - normalized_image.min())
        # 调整大小，常用的是双线性插值
        resized_image = cv2.resize(normalized_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        #反归一化至原灰度范围内
        rescaled_image = cv2.normalize(resized_image, None, min_value, max_value, cv2.NORM_MINMAX, dtype=cv2.CV_16U)
        #转化为与输入矩阵数组一致的格式
        rescaled_image = rescaled_image.astype(dicom_data.pixel_array.dtype)
        # 设置新的像素数据
        dicom_data.PixelData = rescaled_image.tobytes()
        # 更新 Rows 和 Columns 属性
        dicom_data.Rows, dicom_data.Columns = rescaled_image.shape
        #将转化好的图像保存起来
        new_file_path = os.path.join(resized_subfolder_path, dicom_file)
        dicom_data.save_as(new_file_path)
        #保存numpy数组
        dataarray = dicom_data.pixel_array
        dicom_file_name = f"{label}_{dicom_file.split('.')[0]}"
        save_path = os.path.join(numpy_subfolder_path, dicom_file_name)
        np.save(save_path, dataarray)
# 调用函数并指定文件夹路径
for patient_name in patient_names:
    print(patient_name)
    # directory_path为主文件夹（包含所有患者以及各个患者对应序列文件夹）
    input_dicom_folder = os.path.join(directory_path, patient_name)
    resized_folder_path = '/home/data/CNN/127/resizedT1'  # 放置所有患者resized的主文件夹
    numpy_folder_path = '/home/data/CNN/127/resizedT1_numpy'#放置患者numpy数组的文件夹
    #resized_folder_path = '/home/data/CNN/127/resizedT2FS'  # 放置所有患者resized的主文件夹
    #numpy_folder_path = '/home/data/CNN/127/resizedT2FS_numpy'  # 放置患者numpy数组的文件夹
    resized_subfolder_path = os.path.join(resized_folder_path, patient_name)  # 创建resized后的主文件夹下各个患者次级文件夹
    numpy_subfolder_path = os.path.join(numpy_folder_path, patient_name)#存放数据的文件夹
    os.makedirs(resized_subfolder_path, exist_ok=True)
    os.makedirs(numpy_subfolder_path, exist_ok=True)
    new_width = 512
    new_height = 1024
    label = find_value_in_excel(df, patient_name)
    #print(label)
    resize_dicom(input_dicom_folder, resized_subfolder_path, new_width, new_height,numpy_subfolder_path,label)
