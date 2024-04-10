# mp-classifier

## A simple tool to identify a plastic material type by it's Raman spectrum.

An image processing routine normalizes unknown Raman Spectra and a library of reference materials, then matches the unknown spectra to a reference material using the dot product.
Best matches are tabulated into an excel file and processed spectral images are available for each match. 

## How to use it

1) Fetch the repository.
2) Place your data in the 'example' folder, or just use the examples provided
3) Run runme.py

Note: If using your own experimental data it is recommended to collect all spectra between 400 - 2000 cm-1 or larger range. Experimental data should be stored as comma separated value (.csv) files with two unlabeled columns: wavenumber & intensity. Filenames should follow the pattern "[sample name].csv". See "examples" folder for examples.
## Output
A particle library will be saved in the root directory of the repository. This file will be preferentially loaded instead of loading and processing raw files from the library. Other output will be saved in the 'export' folder. Raw and processed composite images are saved for each experimental spectrum overlayed with it's matched library entry. A results spreadsheet is saved which contains the sample name, matching library entry sample name, match score, and dot product.

Match certainty is measured by the dot product of two spectra, which when treated as row vectors, gives a value between 1 (an identical match) and -1 (the vectors are perfectly opposed). For readability, the #dotscore is defined as the inverse cosine of the dot product, resulting in an easily sortable measure between 0 (perfect) and pi (perfectly opposed). In practice, a match score below 1 is considered a good match.

![Example spectral match image](https://github.com/MatthewKowal/mp-classifier/blob/main/readme%20images/example_spec.png)

The above image is a sample output for one experimental image matched to a library spectrum.

## How it works
### Discrete Wavelet Transform (DWT) Noise Reduction
### Iterative DTW Background Estimation and Removal
### Second Derivative Feature Selection
### Mean Centering
### Sum of Squares Normalization


## Microplastic Library
A library of 370+ Raman spectra from 10 different plastic material types. The library contains Raman spectra
collected in Ed Grant's lab in the Chemistry department at the University of British Columbia on an Olympus BX-51
microscope fiber-coupled to a Princeton Instruments Spectrometer. The library was supplemented by [SLoPP and SLoPP-E
Raman Spectral Libraries for
Microplastics Research](https://rochmanlab.wordpress.com/spectral-libraries-for-microplastics-research/) provided by the Rochman Lab at the University of Toronto


| Material | Recycling Code | Library Abbreviation | # of spectra |
| ----------------------------- | :---: | :---:| :---: |
| Acrylonitrile Butadiene Styrene |  ABS | abs  |  50  | 
| Acrylic                         |	     | acr 	|  23  |
| Cellulose                       |	     | ce   |  35  |
|  Polyamide                      |	 PA  | pa   |	 26  |
| Polycarbonate                   |	     | pc 	|  44  |
| Polyethylene                    |	2, 4 | pe 	|  51  |
| Polyethylene Terephthalate      |	  1  | pet 	|  34  |
| Polypropylene                   |	  5  | pp   |	 39  |
| Polystyrene                     |	  6  | ps   |  40  |
| Polyurethane                    |	     | pu   |  12  |
| Polyvinyl Chloride              |	  3  | pvc 	|  18  |

Note: When adding new files to the library, ensure that new library files are organized into folders by material type. Filenames should use the following metadata pattern separated by "_": Material type, Color, Sample name, Source, Status. (e.g. acr__pink__Acrylic 10. Pink Fiber__SloPP__raw.csv, is for an unprocessed (raw) pink acrylic sample found in the SloPP Microplastic dataset 


