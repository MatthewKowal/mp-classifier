# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 18:50:08 2023



Ascii text generated here (using 'Colossal' font)
http://www.patorjk.com/software/taag/

@author: user1
"""


import os
import pandas as pd
import numpy as np
import pywt
import pickle
import matplotlib.pyplot as plt
from collections import Counter


class Spectra:
    def __init__(self, filename, rawdata, category, color, samplename, source, status):
        self.filename  = filename
        self.rawdata    = rawdata
        self.specdata   = pd.DataFrame()
        self.category   = category
        self.color      = color
        self.samplename = samplename
        self.source     = source
        self.status     = status
        self.match      = "no match yet"
        self.dotscore   = 0
        self.dotproduct = 0
        self.matchspec  = pd.DataFrame()
        self.min        = 0
        self.max        = 0
        
        
'''    
888                            888      888      d8b 888                                       
888                            888      888      Y8P 888                                       
888                            888      888          888                                       
888      .d88b.   8888b.   .d88888      888      888 88888b.  888d888 8888b.  888d888 888  888 
888     d88""88b     "88b d88" 888      888      888 888 "88b 888P"      "88b 888P"   888  888 
888     888  888 .d888888 888  888      888      888 888  888 888    .d888888 888     888  888 
888     Y88..88P 888  888 Y88b 888      888      888 888 d88P 888    888  888 888     Y88b 888 
88888888 "Y88P"  "Y888888  "Y88888      88888888 888 88888P"  888    "Y888888 888      "Y88888 
                                                                                           888 
                                                                                      Y8b d88P 
                                                                                       "Y88P"  
'''
#!!!  
def import_raw_library(librarypath):
    '''reads a folder tree of raw plastic Raman standard spectra.
        Then, processed them and saves the data as a pickle file'''
    
    print("Importing Spectra...")
    
    #check for processed library pickle file
    picklefilename = "mp library processed.pickle"
    rootpath, filename = os.path.split(__file__)  # get folder and filename of this script
    pfilepath = os.path.join(rootpath, picklefilename)
    if os.path.exists(pfilepath):
        with open(pfilepath, "rb") as f:
            spectra_list = pickle.load(f)
        print("Loading processed library pickle file...")
        print("\t", len(spectra_list), " files loaded.\n")
        return spectra_list
    
    filelist = get_lib_filelist(librarypath)

    
    
    spectra_list = [] 
    xmin_list = []
    xmax_list = []
    endp_list = []
    
    for f in filelist:
        
        file_name = os.path.basename(f)       
    
        metadata = file_name.split('__')
        if len(metadata)!=5:
            print("ERROR: file (", file_name, ") is missing 1 or more properties")
            print("\t Length of ", len(metadata), " should be 5")
            print("\t", metadata)
        category   = metadata[0] #e.g. pa, ce, pp, pe, etc.
        color      = metadata[1] #e.g. red, green, blue
        samplename = metadata[2] #e.g. Polyamide 1., zoo-23-177
        source     = metadata[3] #e.g. Grant Lab, SLoPP
        status     = metadata[4][:-4] #e.g. raw, preprocessed
                      #metadata[4][:-4] #e.g. raw, preprocessed
        
        #read the csv file from the file list and store it in a pandas dataframe
        #print(a_file)
        rawdata = pd.read_csv(f, names=['cm-1', 'int'])
        rawdata['int'] = rawdata['int'].astype(float)
        
        #tabulate endpoint to later plot in a histogram
        xmin = np.min(rawdata['cm-1'])
        xmax = np.max(rawdata['cm-1'])
        #print(xmin, xmax)
        xmin_list.append(xmin)
        xmax_list.append(xmax)
        endp_list.append(xmin)
        endp_list.append(xmax)
        
        
      
        #make sure the spectra is of even length
        if len(rawdata)%2 == 1: rawdata = rawdata[:-1] #drop last row if odd
    
        #build a new instance of the spectra class with the spectrums name and data and add it to a list
        new_spectra = Spectra(file_name,
                          rawdata,
                          category, color, samplename, source, status)
        spectra_list.append(new_spectra)   
    print("\t", len(spectra_list), " files loaded.\n")
  
    #process spectra
    processed_spectra_list = processSpectra(spectra_list)
    
    #save processed spectra as a pickle file
    save_pickle = True
    if save_pickle == True:
        rootpath, filename = os.path.split(__file__)  # get folder and filename of this script
        #exportpath    = os.path.join(rootpath, "export")
        #if not os.path.exists(exportpath): os.makedirs(exportpath)
        picklepath    = os.path.join(rootpath, "mp library processed.pickle")
        print("saving pickle fils to: ", picklepath)
        pickle.dump(processed_spectra_list, open(picklepath, "wb"))
    
    #save a summary image of all library spectra sorted into categories
    save_libimage = True
    if save_libimage:
        green = "#8945fb"    
        liblist = processed_spectra_list
        
        #get categories
        categories = sorted(list(set([i.category for i in liblist])))
        
        #define plot
        cols = 1
        rows = len(categories)
        fig, ax = plt.subplots(rows,1, figsize=(6,20))

        #plot each category
        for i, cat in enumerate(categories):
            for s in liblist:
                if s.category == cat:
                    ax[i].plot(s.specdata['cm-1'], s.specdata['zeroed'], color=green, alpha=0.1)
                    ax[i].set_title(cat, loc='left')
                    ax[i].set_yticks([])
        plt.tight_layout()

        #save the figure
        rootpath, filename = os.path.split(__file__)  # get folder and filename of this script
        libsumpath    = os.path.join(rootpath, "Library Summary.png")
        plt.savefig(libsumpath)
    
    return processed_spectra_list


def get_lib_filelist(path): #the library files are in subfolders with material names
    filelist = []
    #print(os.listdir(path))
    #print("teststst")
    print("\t class      # of spectra \n\t ------     ---------")
    for folder in os.listdir(path):
        working_path = os.path.join(path, folder)
        #print(working_path)
        #print(os.listdir(working_path))
        n = 0
        for sample in os.listdir(working_path):
            working_file = os.path.join(working_path, sample)
            #print(working_file)
            filelist.append(working_file)
            n+=1
        print("\t ", folder, " \t\t", n)
    return filelist



def get_library_stats(speclist):
    #print(path)
    #spectra_list = pickle.load(open(path, 'rb'))
    cats = [s.category for s in speclist]
    #print(cats)
    
    unique_cats = list(set([s.category for s in speclist]))
    #print(unique_cats)
    # cat_stats = {key: 0 for key in unique_categories}
    
    #tabulate the total number of each plastic category by going through each spectra on the list
    counter = Counter(cats)
    instances = {key: value for key, value in counter.items()}
    #print(instances)
    print("\t class      # of spectra \n\t ------     ---------")
    for key, value in instances.items():
        print(f"\t {key} \t\t {value}")

    
    print("number of spectra: ", len(speclist), "\n wavenumber range: ", speclist[0].min, "-", speclist[0].max)
    
    #base_path = os.path.split(path)[0]
    #save_pickle_path = os.path.join(base_path, "mp library processed infotest.pickle")
    #print(save_pickle_path)
    #print(spectra_list)
    #pickle.dump(spectra_list, open(save_pickle_path, "wb"))
    return len(speclist), speclist[0].min, speclist[0].max





'''
888                            888      888     888          888                                               
888                            888      888     888          888                                               
888                            888      888     888          888                                               
888      .d88b.   8888b.   .d88888      888     888 88888b.  888  888 88888b.   .d88b.  888  888  888 88888b.  
888     d88""88b     "88b d88" 888      888     888 888 "88b 888 .88P 888 "88b d88""88b 888  888  888 888 "88b 
888     888  888 .d888888 888  888      888     888 888  888 888888K  888  888 888  888 888  888  888 888  888 
888     Y88..88P 888  888 Y88b 888      Y88b. .d88P 888  888 888 "88b 888  888 Y88..88P Y88b 888 d88P 888  888 
88888888 "Y88P"  "Y888888  "Y88888       "Y88888P"  888  888 888  888 888  888  "Y88P"   "Y8888888P"  888  888 
                                                                                                               
                                                                                                               
                                                                                                               
8888888b.                                                 .d8888b.                                             
888   Y88b                                               d88P  Y88b                                            
888    888                                               Y88b.                                                 
888   d88P  8888b.  88888b.d88b.   8888b.  88888b.        "Y888b.   88888b.   .d88b.   .d8888b .d8888b         
8888888P"      "88b 888 "888 "88b     "88b 888 "88b          "Y88b. 888 "88b d8P  Y8b d88P"    88K             
888 T88b   .d888888 888  888  888 .d888888 888  888            "888 888  888 88888888 888      "Y8888b.        
888  T88b  888  888 888  888  888 888  888 888  888      Y88b  d88P 888 d88P Y8b.     Y88b.         X88        
888   T88b "Y888888 888  888  888 "Y888888 888  888       "Y8888P"  88888P"   "Y8888   "Y8888P  88888P'        
                                                                    888                                        
                                                                    888                                        
                                                                    888                                      
'''
#!!!

def import_expdata(exp_path):
    
    #import experimental data
    exp_data = [] # a list to hold all the new spectra
    filelist = get_exp_filelist(exp_path) # a list of all the new spectra
    
    #import each new spectrum
    for a_file in filelist:
        #determine metadate
        filename = os.path.basename(a_file)
        category   = "unknown"                       #metadata[0] #e.g. pa, ce, pp, pe, etc.
        color      = "not specified"                 #metadata[1] #e.g. red, green, blue
        samplename = filename[:-4]                   #metadata[2] #e.g. Polyamide 1., zoo-23-177
        source     = "Grant Lab Experimental Sample" #metadata[3] #e.g. Grant Lab, SLoPP
        status     = "raw" 
        #read spectral data
        rawdata = pd.read_csv(a_file, names=['cm-1', 'int'])
        rawdata['int'] = rawdata['int'].astype(float)
        #make sure the spectra is of even length
        if len(rawdata)%2 == 1: rawdata = rawdata[:-1] #drop last row if odd
        #build a new instance of the spectra class with the spectrums name and data and add it to a list
        new_spectra = Spectra(filename, rawdata, category, color, samplename, source, status)
        exp_data.append(new_spectra)
        #exp_data is a list of Spectra that conotains all our experimental data
    
    #process spectra
    processed_expdata = processSpectra(exp_data)


    return processed_expdata#a_string

def get_exp_filelist(path): #experimental spetra are all in one folder
    print("\n\nLoading experiment files...")
    print("\t\t NOTE: all files should be in the following format: [sample name].csv ")
    print("\t\t       Be sure to use brackets!")
    filelist = []
    for filename in os.listdir(path):
        full_filepath = os.path.join(path, filename)
        filelist.append(full_filepath)
        print(filename)
    print("\n")
    return filelist

def dotScore(v1, v2):
    
    len1 = np.sqrt(np.dot(v1, v1))
    len2 = np.sqrt(np.dot(v2, v2))
    ds = np.arccos(np.dot(v1, v2) / (len1 * len2))
    
    return ds

def dotProduct(v1, v2):
    
    len1 = np.sqrt(np.dot(v1, v1))
    len2 = np.sqrt(np.dot(v2, v2))
    dp = np.dot(v1, v2) / (len1 * len2)
    #print(dp)
    return np.cos(dp)



def make_equal_spec_length(exp, lib):
    ''' this function takes in two spectra object of arbitrary length
    and find the overlapping portion and then makes truncated versions
    of that section. The function then returns the two equal length spectra
    as np.arrays '''
    
    col= 'contodo' #this is the column in the data we will use
    emin, emax = np.min(exp.specdata["cm-1"]), np.max(exp.specdata["cm-1"])
    lmin, lmax = np.min(lib.specdata["cm-1"]), np.max(lib.specdata["cm-1"])
    dif1 = emin - lmin
    dif2 = emax - lmax
    
    expfront, libfront = 0, 0
    expback, libback   = -1, -1
    
    if dif1 > 0: libfront = int(dif1)
    if dif1 < 0: expfront = int(dif1 * -1) 
    
    if dif2 > 0: expback  = int(dif2 * -1) -1
    if dif2 < 0: libback  = int(dif2) -1
    
    xdata = np.array(exp.specdata[col])
    ldata = np.array(lib.specdata[col])
    xdata_ = xdata[expfront:expback]
    ldata_ = ldata[libfront:libback]
    
    # print("\n exp", emin, emax, "\n lib", lmin, lmax, "\n difs", dif1, dif2)
    # print("original lengths:", len(xdata), len(ldata))
    # print("exp trim:", expfront, expback, "\nlib trim:", libfront, libback)
    # print(type(xdata), type(ldata))
    # print("new lengths:", len(xdata_), len(ldata_))
    return xdata_, ldata_


def matchSpectra(exp_list, lib_list):
    matched_exp_list = []
    for exp in exp_list:  #loop through each experimental data
        scores = []
        dotproducts = []
        for lib in lib_list:
            xdata_, ldata_ = make_equal_spec_length(exp, lib)
            scores.append(dotScore(xdata_,ldata_)) 
            dotproducts.append(dotProduct(xdata_, ldata_))
        best_match_idx = np.argmin(scores)       
        exp.match      = lib_list[best_match_idx].category
        exp.dotscore   = scores[best_match_idx]
        exp.dotproduct = dotproducts[best_match_idx]
        exp.matchspec  = lib_list[best_match_idx].specdata        
        matched_exp_list.append(exp)  
    return matched_exp_list




'''
 .d8888b.  d8b                            888                                       
d88P  Y88b Y8P                            888                                       
Y88b.                                     888                                       
 "Y888b.   888  .d88b.  88888b.   8888b.  888                                       
    "Y88b. 888 d88P"88b 888 "88b     "88b 888                                       
      "888 888 888  888 888  888 .d888888 888                                       
Y88b  d88P 888 Y88b 888 888  888 888  888 888                                       
 "Y8888P"  888  "Y88888 888  888 "Y888888 888                                       
                    888                                                             
               Y8b d88P                                                             
                "Y88P"                                                              
8888888b.                                                     d8b                   
888   Y88b                                                    Y8P                   
888    888                                                                          
888   d88P 888d888 .d88b.   .d8888b .d88b.  .d8888b  .d8888b  888 88888b.   .d88b.  
8888888P"  888P"  d88""88b d88P"   d8P  Y8b 88K      88K      888 888 "88b d88P"88b 
888        888    888  888 888     88888888 "Y8888b. "Y8888b. 888 888  888 888  888 
888        888    Y88..88P Y88b.   Y8b.          X88      X88 888 888  888 Y88b 888 
888        888     "Y88P"   "Y8888P "Y8888   88888P'  88888P' 888 888  888  "Y88888 
                                                                                888 
                                                                           Y8b d88P 
                                                                            "Y88P"  
8888888b.                   888    d8b                                              
888   Y88b                  888    Y8P                                              
888    888                  888                                                     
888   d88P .d88b.  888  888 888888 888 88888b.   .d88b.  .d8888b                    
8888888P" d88""88b 888  888 888    888 888 "88b d8P  Y8b 88K                        
888 T88b  888  888 888  888 888    888 888  888 88888888 "Y8888b.                   
888  T88b Y88..88P Y88b 888 Y88b.  888 888  888 Y8b.          X88                   
888   T88b "Y88P"   "Y88888  "Y888 888 888  888  "Y8888   88888P'    

'''
#!!!

def processSpectra(spectra_list):
    ''' This function takes in a list of Sectra Objects containing
    raw raman spectral data. The data is processed using a variety of techniques
    including DTW, 2nd Derivative, Mean-centering, and Sum-of-Squares Normalization.
    An updated list of Spectra Objects containing processed data is then returned.
    
    I somewhat regret using Pandas for this, but here it is...
    
    Each "column" contains the following data:
    -----------------------------------------
    cm-1         Wavenumbers
    int          Raw, unprocessed Raman intensity
    grad         DWT-Lowpass, Gradient, Mean-center, L2-Normalize
    contodo      DWT-Lowpass, DWT-Background-correction, Gradient, Mean-center, L2-Normalize
    zeroed       DWT-Lowpass, DWT-Background-correction, L2-Normalize, ?why not zero the bg?
    
    Anecdotally, contodo to works the best for spectral matching, but I suspect
    grad should work equally well because there shouldnt really be a need for 
    estimating and removing an extremely low frequency background (fluoresence)
    when a second derivative will do about the same thing either way.
    
    
    '''
    
    PRINT_STUFF = False
    print("Processing data...\n\t-Resampling...\n\tCalculating Gradient...\n\t",
          "Calculating Zero-Baseline...\n\tetc...")
    for c, s in enumerate(spectra_list):
        #print("WORKING ON            ", s.filename)
        #print(a_spectra.status)
        if s.status == "processed":
            print("write code to handle data thats already been processed")
        elif s.status == "raw":
            #print("processing raw data...")
            #define the beginning and end point as integers
            start = int(s.rawdata['cm-1'].min()+1)#.astype(int)
            stop = int(s.rawdata['cm-1'].max())#.astype(int)
            #print("\nstart",start,"\nstop",stop)
            if (stop-start)%2 == 1:  stop-=1 #make sure spectrum is even
            num=stop-start

            # s.min = rangeMin
            # s.max = rangeMax
            
            ''' Resample the Spectrum '''
            #print(s.rawdata)
            #assign the cm-1 columns to be the index of the raw data
            s.rawdata.index = s.rawdata['cm-1']
            #create a new regular index
            reg_idx = np.linspace(start, (stop-1), num)
            #create a new dataframe for the newly indexed, blank data
            reg_idx_data = pd.DataFrame(data=[],
                                        index=reg_idx,
                                        columns=["int"])
            #add the new blank dataframe to the original dataframe, then sort it by index
            upsampled_spectrum = pd.concat([s.rawdata, reg_idx_data]).sort_index()
            #print(upsampled_spectrum)
            #interpolate the data and fill it in relative to the index and drop duplicates
            #print(upsampled_spectrum.loc[402:])
            #print(type(upsampled_spectrum))
            interpolated_spectrum = upsampled_spectrum.interpolate(method='index').drop_duplicates()
            #print(interpolated_spectrum.loc[402:])
            #resample the data by saving only the data at the disired index
            resampled_spectrum = interpolated_spectrum.reindex(reg_idx)
            #print(resampled_spectrum)
            #update the spectra object
            s.specdata = resampled_spectrum
            #print(resampled_spectrum)
            

            ''' Flatten the first and last 50 data points '''
            trimlen = 50
            #print(type(s.specdata))
            #print(s.specdata['int'].iloc[trimlen])
            s.specdata['int'].iloc[:trimlen] = s.specdata['int'].iloc[trimlen].copy() #flatten first n datapoints
            
            ''' Method 0: lowpass, 2nd deriv, mean-center, l2 norm '''
            #print("print the type: ", type(spec[i].data['int']))
            #gradient & mean centered spectrum
            s.specdata['grad'] =        lowpass(s.specdata['int'].copy(), 3)
            #s.specdata['grad'] = removebaseline(s.specdata['grad'].copy(), 7, 10)
            s.specdata['grad'] =    np.gradient(s.specdata['grad'].copy(), edge_order=2)
            s.specdata['grad'] =                s.specdata['grad'] - np.mean(s.specdata['grad'].copy())
            s.specdata['grad'] =      normalize(s.specdata['grad'].copy())
            #print(np.sum(np.square(s.specdata['grad'])))        
            
            ''' Method 1: lowpass, dwt-bg, 2nd deriv, mean-center, l2-norm'''
            #gradient & mean centered spectrum
            s.specdata['contodo'] =        lowpass(s.specdata['int'].copy(), 3)
            s.specdata['contodo'] = removebaseline(s.specdata['contodo'].copy(), 7, 10)
            s.specdata['contodo'] =    np.gradient(s.specdata['contodo'].copy(), edge_order=2)
            s.specdata['contodo'] =                s.specdata['contodo'] - np.mean(s.specdata['grad'].copy())
            s.specdata['contodo'] =      normalize(s.specdata['contodo'].copy())
            #print(np.sum(np.square(s.specdata['grad'])))        
            
            
            ''' Method 2: low pass, dwt-bg, l2-norm'''        
            #zero baseline spectrum
            s.specdata['zeroed'] =        lowpass(s.specdata['int'].copy(), 3)
            s.specdata['zeroed'] = removebaseline(s.specdata['zeroed'].copy(), 7, 10)
            s.specdata['zeroed'] =      normalize(s.specdata['zeroed'].copy())
            #print(spec[i].data['zeroed'].index)
            
            
         
            #update status
            if PRINT_STUFF: print(s.filename, "...Done")
            s.status = "processed"
    print("\t\t", c, "of", len(spectra_list), "files processed.")
           
    return spectra_list

def normalize(spectra):
    '''spectra is a single pandas series representing one spectrum'''

    #spectra -= np.min(spectra)
    sumofsquares = np.sum(np.square(spectra)) #compute sum of squares
    newspec = spectra/np.sqrt(sumofsquares)  #normalize to sum of squares
    
    return newspec

def lowpass(spectra, level): 
    '''spectra is a single pandas series representing one spectrum'''
    
    #unpack spectra from dataframe
    #rawdata = spectra#["int"]
    
    #first decompose to spectra to the specified level.
    coeffs = pywt.wavedec(spectra, 'sym5', level=level) #tranform to wavelets
    
    #remove all high frequency levels
    for i in range(1,(level+1)):
        coeffs[i] = np.zeros_like(coeffs[i])
        #print(i)
    
    #reconstruct the wavelets to spectra space
    newspec = pywt.waverec(coeffs, 'sym5')
    
    
    #save data in a new column in dataframe
    #spectra["int2"] = newspec
    return newspec#spectra
    
def removebaseline(spectra, level, iterations):
    '''spectra is a single pandas series representing one spectrum'''
    #rawdata = spectra#["int2"]  #unpack spectra from the dataframe
    bg = spectra#rawdata  #the background will be calculated from the spectra
    for i in range(iterations):
        
        #first decompose to spectra to the specified level.
        coeffs = pywt.wavedec(bg, 'sym5', level=level) #tranform to wavelets
        
        #remove all high frequency levels
        for i in range(2,(level+1)):
            coeffs[i] = np.zeros_like(coeffs[i])
            #print(i)
        
        #reconstruct the wavelets to spectra space
        low_freq_spec = pywt.waverec(coeffs, 'sym5')
        
        #take the mimimum of current spec background vs low freq reconstruction
        bg = np.minimum(bg, low_freq_spec)
        
    #subtract the background from the original spectra
    newspec = spectra - bg#rawdata - bg
    
    #spectra["int3"] = newspec
    return newspec#spectra


'''
 .d8888b.                                             888            
d88P  Y88b                                            888            
888    888                                            888            
888         .d88b.  88888b.   .d88b.  888d888 8888b.  888888 .d88b.  
888  88888 d8P  Y8b 888 "88b d8P  Y8b 888P"      "88b 888   d8P  Y8b 
888    888 88888888 888  888 88888888 888    .d888888 888   88888888 
Y88b  d88P Y8b.     888  888 Y8b.     888    888  888 Y88b. Y8b.     
 "Y8888P88  "Y8888  888  888  "Y8888  888    "Y888888  "Y888 "Y8888  
                                                                     
                                                                     
                                                                     
8888888b.                                    888                     
888   Y88b                                   888                     
888    888                                   888                     
888   d88P .d88b.  88888b.   .d88b.  888d888 888888 .d8888b          
8888888P" d8P  Y8b 888 "88b d88""88b 888P"   888    88K              
888 T88b  88888888 888  888 888  888 888     888    "Y8888b.         
888  T88b Y8b.     888 d88P Y88..88P 888     Y88b.       X88         
888   T88b "Y8888  88888P"   "Y88P"  888      "Y888  88888P'         
                   888                                               
                   888                                               
                   888                                               
'''                                        
#!!!

def export_excel(matched_speclist):
    print("\nExporting excel file")
    #load matched data
    #m_exp_data =  pickle.load(open(m_exp_data_pkl_path, 'rb'))
    m_exp_data = matched_speclist
    #create blank excel file
    output = pd.DataFrame(columns=[])
    
    # #fill in excel file
    # for s in m_exp_data:
    #     output = output.append({"sample name":s.samplename,
    #                             "prediction":s.match,
    #                             "match score":s.dotscore,
    #                             }, ignore_index=True)  
    for s in matched_speclist:
        new_row = {"sample name":s.samplename,
                   "prediction":s.match,
                   "match score":s.dotscore,
                   "dot product":s.dotproduct}
        #print(type(new_row))
        #print(new_row)
        new_row_df = pd.DataFrame(new_row, index=[0])

        old_rows = output.copy()
        output = pd.concat([old_rows, new_row_df], ignore_index=True)


    #generate filename
    rootpath, filename = os.path.split(__file__)  # get folder and filename of this script
    exportpath    = os.path.join(rootpath, "export")
    if not os.path.exists(exportpath): os.makedirs(exportpath)
    save_excel_path = os.path.join(exportpath, "_matched_spectra.xlsx")
    
    #save excel file
    output.to_excel(save_excel_path, index=False)
    
    
    # a_string  = "Saved Match Data to:\n"
    # a_string += "\t\t" + save_excel_path +"\n\n"
    # a_string += "Results:\n"
    # a_string += "\n"+ output.to_string() +"\n"
    #a_string = m_exp_data_pkl_path+"\n"+base_path+"\n"+save_filename+"\n"+save_excel_path
    #a_string += "\n"+ output.to_string()
    
    return #a_string


def export_images(matched_speclist): #export images
    print("saving images")
    col="contodo"
    # base_path = os.path.split(mpath)[0]
    # expdata_matches = pickle.load(open(mpath, 'rb'))
    
    # a_string = "Saving composite spectra images:\n"
    
    #generate filepath
    rootpath, filename = os.path.split(__file__)  # get folder and filename of this script
    exportpath    = os.path.join(rootpath, "export")
    if not os.path.exists(exportpath): os.makedirs(exportpath)

    
    for s in matched_speclist:
        img_filepath = os.path.join(exportpath, (s.samplename + "-spec.png"))
    
        # plt.figure(figsize=(8,8))
        # plt.suptitle(s.samplename)
    
        # plt.subplot(2,1,1)
        # #plot unprocessed experimental data
        # plt.plot(s.specdata["cm-1"], s.specdata["int"],
        #           color="#002145",
        #           label=("Experimental Spectra, Unprocessed"))
        # #plot unprocessed library match
        # plt.plot(s.matchspec["cm-1"], s.matchspec["int"],
        #            color="#00A7E1",
        #            label=("Library Spectra, Unprocessed"))
        # plt.legend()
    
        # plt.subplot(2,1,2)
        # # #plot baseline corrected match
        # plt.plot(s.specdata["cm-1"], s.specdata[col],
        #           color="#002145",
        #           label=("Experimental Spectra, Processed"))
        # # #plot baseline corrected sample
        # plt.plot(s.matchspec["cm-1"], s.matchspec[col],
        #           color="#00A7E1",
        #           label=("Library Spectra, Processed: " + s.match))
                  
        plt.figure(figsize=(12,8))
        plt.suptitle((s.samplename+" - dotproduct:", str(s.dotproduct)))
    
        plt.subplot(3,1,1)
        #plot unprocessed experimental data
        plt.plot(s.specdata["cm-1"], s.specdata["int"],
                  color="#002145",
                  label=("Experimental Spectra, Unprocessed"))
        #plot unprocessed library match
        plt.plot(s.matchspec["cm-1"], s.matchspec["int"],
                   color="#00A7E1",
                   label=("Library Spectra, Unprocessed"))
        plt.legend()
    
        plt.subplot(3,1,2)
        #plot unprocessed experimental data
        plt.plot(s.specdata["cm-1"], s.specdata["zeroed"],
                  color="#002145",
                  label=("Experimental Spectra, Baseline Corrected"))
        #plot unprocessed library match
        plt.plot(s.matchspec["cm-1"], s.matchspec["zeroed"],
                   color="#00A7E1",
                   label=("Library Spectra, Baseline Corrected"))
        plt.legend()
    
    
        plt.subplot(3,1,3)
        # #plot baseline corrected match
        plt.plot(s.specdata["cm-1"], s.specdata[col],
                  color="#002145",
                  label=("Experimental Spectra, Processed"))
        # #plot baseline corrected sample
        plt.plot(s.matchspec["cm-1"], s.matchspec[col],
                  color="#00A7E1",
                  label=("Library Spectra, Processed: " + s.match))
        
    
    
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(img_filepath)#, bbox_inches="tight")
        #plt.show()
        plt.close()
        #a_string += "\t\t" + img_filepath + "\n"
        
    #a_string += "\n"    
    #a_string = "test" + mpath + "\n" + base_path + "\n" + img_filepath
    
    return #a_string
                            

              
import numpy as np                         
def export_csv(matched_spectra): #export images
    print("saving excel files")
    
    #generate filepath
    rootpath, filename = os.path.split(__file__)  # get folder and filename of this script
    exportpath    = os.path.join(rootpath, "export-csv")
    if not os.path.exists(exportpath): os.makedirs(exportpath)
    
    
    for s in matched_spectra:
        
        
        #save experiment spectrum
        excel_filepath = os.path.join(exportpath, (s.samplename + ".xlsx"))
        col1 = np.array(s.specdata["cm-1"])
        col2 = np.array(s.specdata["zeroed"])
        col3 = np.array(s.matchspec["cm-1"])
        col4 = np.array(s.matchspec["zeroed"]) 
        #exceldata = np.transpose([col1, col2, col3, col4])
        exceldata = zip(col1, col2, col3, col4)
        column_names = ["Sample",s.samplename,"Library",s.match]
        
        df = pd.DataFrame(exceldata, columns=column_names)
        df.to_excel(excel_filepath, index=False)
    
    return #a_string
                                       