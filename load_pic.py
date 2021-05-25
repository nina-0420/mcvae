# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 19:58:45 2021

@author: scnc
"""

import nibabel as nib
import matplotlib.pyplot as plt


img_arr = nib.load('./input_data/debug_loaders/sax_generated_1419771_21016_1_0.nii.gz').get_fdata()
ing_arr1=img_arr[10,:,:]
plt.imshow(ing_arr1)
plt.pause(3)
