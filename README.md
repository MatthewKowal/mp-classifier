# mp-classifier

### A simple tool to identify a plastic material type by it's Raman spectrum.

An image processing routine normalizes unknown Raman Spectra and a library of reference materials, then matches the unknown spectra to a reference material using the dot product.
Best matches are tabulated into an excel file and processed spectral images are available for each match. 

### How to use it

Fetch the repository and run runme.py

### How it works
All of the signal processing routines are written in mp

### Microplastic Library
A library of 370+ Raman spectra from 10 different plastic material types. The library contains Raman spectra
collected in Ed Grant's lab in the Chemistry department at the University of British Columbia on an Olympus BX-51
microscope fiber-coupled to a Princeton Instruments Spectrometer. The library was supplemented by [SLoPP and SLoPP-E
Raman Spectral Libraries for
Microplastics Research](https://rochmanlab.wordpress.com/spectral-libraries-for-microplastics-research/) provided by the Rochman Lab at the University of Toronto

| material | # of spectra |
| -------- | ------------ | 
| abs      |    50| 
|	 acr 	   |    23|
|	 ce   	 |    35|
|	 pa      |	  26|
|	 pc 	   | 	  44|
|	 pe 	   |    51|
|	 pet 	   | 	  34|
|	 pp      |	  39|
|	 ps      |    40|
|	 pu 	   |    12|
|	 pvc 	   |    18|

## Tips for use
- It is recommended to collect all spectra between 400 - 2000 cm-1 or larger range.
- A match score below 1 is considered "good".
- Experimental .csv data should be named with the sample name surrounded by brackets and ending in .csv (e.x. "[sample name].csv"). The file should contain two columns (wavenumber, intensity)
- Library entries should be placed in a folder according to its material type. Each file should be named in the following format "material_color_samplename_origin_status", where material is am abbreviated form of the plastic type, color is the color of the sample, origin is the place where the spectrum was collected(i.e. "grant lab"), and status is either "raw" or "processed"
