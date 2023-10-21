# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 18:31:06 2023

@author: user1
"""

import os
import sys
import pickle

folder, filename = os.path.split(__file__)  # get folder and filename of this script
#modfolder = os.path.join(folder)            # I cant remember why this is here
sys.path.insert(0, folder)               # add the current folder to the system path

import mp_backend as mp
from mp_backend import Spectra




''' get the location of the current script'''
script_path      = os.path.abspath(__file__)       # Use __file__ to get the path of the currently running script
script_directory = os.path.dirname(script_path)    # Use os.path.dirname to get the directory containing the script
print("Directory of the currently running script:", script_directory)

''' define the endpoint cutoff for all spectra. '''
wavenum_min = 402
wavenum_max = 1990

''' define location of the the followinf files '''
lib_path = os.path.join(script_directory, "mp library") #microplastics library database, separated into material categories
lib_pkl  = os.path.join(script_directory, "mp library processed.pickle")  # a filename to load / save the library when its pickled
exp_path = os.path.join(script_directory, "examples")                     # this is where you put files to be processed
output_dir = os.path.join(script_directory, "output")               # this is where generated reports and images will be saved

if not os.path.exists(output_dir): os.makedirs(output_dir) # if path doesnt exist then create it now


''' load a previously processed library, or load and save a new libary from a new library directory '''
try:
    print("\n Looking for: ", lib_pkl)
    lib_list      = pickle.load(open(lib_pkl, "rb"))
    print("pickle load successful!")
except FileNotFoundError:
    print("\n Could not find pickle library file. Importing and processing new spectra from \n\t\t", lib_path)
    lib_list = mp.import_raw_library(lib_path, wavenum_min, wavenum_max, save_pickle=True)
mp.get_library_stats(lib_list)



''' match the new experimental spectra to the library spectra '''
matched_speclist = mp.lpmp_exp_data(exp_path, lib_list, wavenum_min, wavenum_max)


''' generate spreadsheet with match scores ( less than 1 is good) '''
mp.export_excel(matched_speclist, output_dir)


''' generate images '''
mp.export_images(matched_speclist, output_dir)
