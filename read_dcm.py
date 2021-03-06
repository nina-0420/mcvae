# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 11:05:22 2021

@author: scnc
"""

from os.path import dirname, join
from pprint import pprint
import pydicom
from pydicom.filereader import read_dicomdir
import os
import matplotlib.pyplot as plt

# =============================================================================
# path = (r'C:/Users/scnc/Downloads/code/MI_pred_mcvae_ukbb-master/data-tagging/3721418/image/time001/DICOMDIR')
# #load the data 
# dicom_dir = read_dicomdir(path)
# base_dir = dirname(path)
# 
# #go through the patient record and print information
# for patient_record in dicom_dir.patient_records:
#     if (hasattr(patient_record, 'PatientID') and
#             hasattr(patient_record, 'PatientName')):
#         print("Patient: {}: {}".format(patient_record.PatientID,
#                                        patient_record.PatientName))
#     studies = patient_record.children
#     # got through each serie
#     for study in studies:
#         print(" " * 4 + "Study {}: {}: {}".format(study.StudyID,
#                                                   study.StudyDate,
#                                                   study.StudyDescription))
#         all_series = study.children
#         # go through each serie
#         for series in all_series:
#             image_count = len(series.children)
#             plural = ('', 's')[image_count > 1]
# 
#             # Write basic series info and image count
# 
#             # Put N/A in if no Series Description
#             if 'SeriesDescription' not in series:
#                 series.SeriesDescription = "N/A"
#             print(" " * 8 + "Series {}: {}: {} ({} image{})".format(
#                 series.SeriesNumber, series.Modality, series.SeriesDescription,
#                 image_count, plural))
# 
#             # Open and read something from each image, for demonstration
#             # purposes. For simple quick overview of DICOMDIR, leave the
#             # following out
#             print(" " * 12 + "Reading images...")
#             image_records = series.children
#             image_filenames = [join(base_dir, *image_rec.ReferencedFileID)
#                                for image_rec in image_records]
# 
#             datasets = [pydicom.dcmread(image_filename)
#                         for image_filename in image_filenames]
# 
#             patient_names = set(ds.PatientName for ds in datasets)
#             patient_IDs = set(ds.PatientID for ds in datasets)
# 
#             # List the image filenames
#             print("\n" + " " * 12 + "Image filenames:")
#             print(" " * 12, end=' ')
#             pprint(image_filenames, indent=12)
# 
#             # Expect all images to have same patient name, id
#             # Show the set of all names, IDs found (should each have one)
#             print(" " * 12 + "Patient Names in images..: {}".format(
#                 patient_names))
#             print(" " * 12 + "Patient IDs in images..: {}".format(
#                 patient_IDs))
# =============================================================================
            

ds = pydicom.read_file(r'C:/Users/scnc/Downloads/code/MI_pred_mcvae_ukbb-master/data-tagging/3721418/image/time001/DICOMDIR')
pixel_data = list()
for record in ds.DirectoryRecordSequence:
        if record.DirectoryRecordType == "IMAGE":
        # Extract the relative path to the DICOM file
            path = os.path.join(*record.ReferencedFileID)
            dcm = pydicom.read_file(path)

            # Now get your image data
            pixel_data.append(dcm.pixel_array)
            
            