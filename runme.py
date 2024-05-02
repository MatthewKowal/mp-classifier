# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 23:27:21 2024

@author: user1
"""

#this is for importing the mp_backend local file
import os
import sys
folder, filename = os.path.split(__file__)  # get folder and filename of this script
#modfolder = os.path.join(folder)            # I cant remember why this is here
sys.path.insert(0, folder)               # add the current folder to the system path
import mp_backend as mp




rootpath, filename = os.path.split(__file__)  # get folder and filename of this script
librarypath    = os.path.join(rootpath, "mp library")
experimentpath = os.path.join(rootpath, "examples")
exporttpath    = os.path.join(rootpath, "export")



#import and process library
liblist = mp.import_raw_library(librarypath)

    



#import and process experimental data
explist = mp.import_expdata(experimentpath)


#match exp data to library data
matched_spectra = mp.matchSpectra(explist, liblist)


#print output
mp.export_excel(matched_spectra)
mp.export_images(matched_spectra)