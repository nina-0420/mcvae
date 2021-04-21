



import os, glob
import pydicom as pyd
import matplotlib.pyplot as plt
import pandas as pd
from pydicom import dcmread
import pylibjpeg
import nibabel as nib
import numpy as np
import os
import dicom
import SimpleITK as sitk


def save_nii(filename, data, pixel_spacing, thickness):
    
    dataITK = sitk.GetImageFromArray(data)
    dataITK.SetSpacing([pixel_spacing[0],pixel_spacing[1], thickness])
    sitk.WriteImage(dataITK, filename)


def dcm2nii_2(path, pat, mod, center):
    path = path

    #
    filenames = os.listdir(path)
    new_filenames = []
    for i in range(0, len(filenames)):
        if filenames[i].split('.')[-1] == 'dcm':
            new_filenames.append(filenames[i])
    filenames = new_filenames
    filenames.sort(key=lambda x: int(x.split('.')[-2]))
    # print(filenames)


    ds = pyd.dcmread(path + filenames[0])
    image = ds.pixel_array

    
    print(np.shape(np.shape(image))[0])
    if np.shape(np.shape(image))[0] == 2:

        pixel_spacing = ds.PixelSpacing
        thickness = ds.SliceThickness
        print('pixel spacing: ', pixel_spacing, '  thickness: ', thickness)


        #
        file_path = path
        series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(file_path)
        nb_series = len(series_IDs)
        print(nb_series)
        series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(file_path)
        series_reader = sitk.ImageSeriesReader()
        series_reader.SetFileNames(series_file_names)
        image3D = series_reader.Execute()
        print(image3D.GetSize())

        filename = 'E:/s3/nifti-sth/' + center + '{}_{}.nii.gz'.format(pat, mod)
        sitk.WriteImage(image3D, filename)
        #

    else:
        print('cannot solve compressed dicom')



def dcm2nii(path, pat, mod, center):

    filenames = os.listdir(path)
    ds = pyd.dcmread(path + filenames[0])
    image = ds.pixel_array
    print(np.shape(np.shape(image))[0])
    if np.shape(np.shape(image))[0] == 2:
        #
        file_path = path
        series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(file_path)
        nb_series = len(series_IDs)
        print(nb_series)
        series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(file_path)
        series_reader = sitk.ImageSeriesReader()
        series_reader.SetFileNames(series_file_names)
        image3D = series_reader.Execute()
        print(image3D.GetSize())

        filename = 'E:/s3/nifti/' + center + '{}_{}.nii.gz'.format(pat, mod)
        sitk.WriteImage(image3D, filename)
    else:
        print('cannot solve compressed dicom')


def if_XRA(folder_path):
    XRA = 'XA'
    CTA = 'CT'
    MRA = 'MR'
    files = os.listdir(folder_path)
    print(len(files))
    img_path = folder_path + files[0]
    # print('img_path', img_path)
    image = pyd.dcmread(img_path)
    if image.Modality == XRA:
        c = True
    else:
        c = False

    return c



if __name__ == '__main__':

    root_dir = 'E:/s3/dataset-aneurist-ais/AIS/'
    folder1 = 'CHILE/'
    folder2 = 'STH/'
    folder3 = 'UNIGE/'
    folder4 = 'UPF/'

    chile_path = root_dir + folder1
    sth_path = root_dir + folder2
    unige_path = root_dir + folder3
    upf_path = root_dir + folder4

    pat_dir = os.listdir(sth_path)


    # for i in range(0, len(pat_dir)):
    for i in range(0, len(pat_dir)):
        print('############# Start #############')
        mod_dir = os.listdir(sth_path + pat_dir[i] + '/')
        print(pat_dir[i])
        # print(len(pat_dir))
        for j in range(0, len(mod_dir)):

            # print('############# Start #############')
            print(i, '-', j)
            # print(mod_dir[j])
            # print(len(mod_dir))

            if_mod = if_XRA(sth_path + pat_dir[i] + '/' + mod_dir[j] + '/')
            dcm2nii_path = sth_path + pat_dir[i] + '/' + mod_dir[j] + '/'
            print(if_mod)
            # print('##################################')

            if if_mod:
                dcm2nii_2(path=dcm2nii_path, pat=i, mod=j, center='sth')



