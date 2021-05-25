# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 22:27:03 2021

@author: scnc
"""
#查看和显示nii.gz文件
 
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pylab as plt
import nibabel as nib
from nibabel import nifti1
from nibabel.viewers import OrthoSlicer3D
 
example_filename = 'C:/Users/scnc/Downloads/code/tagging_cine_exp/results/2021-04-29_21-58-15/gen_data/reconstructed_1001629/1.3.12.2.1107.5.2.18.141243.2018051910421789022064160.nii.gz'
img = nib.load(example_filename)
print(img)
print(img.header['db_name'])  # 输出头信息
 
#shape有四个参数 patient001_4d.nii.gz
#shape有三个参数 patient001_frame01.nii.gz   patient001_frame12.nii.gz
#shape有三个参数  patient001_frame01_gt.nii.gz   patient001_frame12_gt.nii.gz
width, height, queue = img.dataobj.shape
OrthoSlicer3D(img.dataobj).show()
 
num = 1
for i in range(0, queue, 10):
    img_arr = img.dataobj[:, :, i]
    plt.subplot(5, 4, num)
    plt.imshow(img_arr, cmap='gray')
    num += 1
 
plt.show()