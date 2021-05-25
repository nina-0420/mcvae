# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 20:16:54 2021

@author: scnc
"""
#pip install SimpleItk        # 读取CT医学图像
#pip install tqdm             # 可扩展的Python进度条，封装迭代器
#pip install pydicom          # 用于读取 dicom 图片
#pip install opencv-python
import os
import SimpleITK
import numpy as np
import cv2
from tqdm import tqdm
import shutil
 
def convert_from_dicom_to_jpg(img,low_window,high_window,save_path):
    lungwin = np.array([low_window*1.,high_window*1.])
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])    #归一化
    newimg = (newimg*255).astype('uint8')                #将像素值扩展到[0,255]
    stacked_img = np.stack((newimg,) * 3, axis=-1)
    cv2.imwrite(save_path, stacked_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
 
if __name__ == '__main__':
	#dicom文件目录
    dicom_dir = "/content/drive/My Drive/MI_pred_mcvae_ukbb-master/tagging_trans/10xxxxx/1005742/"
 
    path = "dcm_1_png"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    for i in tqdm(os.listdir(dicom_dir)):
        dcm_image_path = os.path.join(dicom_dir,i)  # 读取dicom文件
        name, _ = os.path.splitext(i)
        output_jpg_path = os.path.join(path, name+'.png')
        ds_array = SimpleITK.ReadImage(dcm_image_path)  # 读取dicom文件的相关信息
        img_array = SimpleITK.GetArrayFromImage(ds_array)  # 获取array
        # SimpleITK读取的图像数据的坐标顺序为zyx，即从多少张切片到单张切片的宽和高，此处我们读取单张，因此img_array的shape
        # 类似于 （1，height，width）的形式
        shape = img_array.shape
        img_array = np.reshape(img_array, (shape[1], shape[2]))  # 获取array中的height和width
        high = np.max(img_array)
        low = np.min(img_array)
        convert_from_dicom_to_jpg(img_array, low, high, output_jpg_path)  # 调用函数，转换成jpg文件并保存到对应的路径