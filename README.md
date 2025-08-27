bitext-preprocessor.py is designed to take in NTs in 3 different languages. The input files need to be usfm titled 41.usfm-67.usfm. 
The output is a 3-language "bi-text", a tritext? A series of CSV files (41-47) with index locations.
Index locations are defined by biblical 8 digit. Matthew 2:3 is 41002003. First two digits identify the book, the next three the chapter, and the final three the verse. 

usfm_to_clean_txt.py is intended to take files 41.usfm to 67.usfm and convert them to one line for each verse in a .txt file. 
Cleaning involves removing everything except letters and white spaces. Converted to all lowercase as well. 


usfm_to_trigram.py is intended to take a usfm file and convert it to a trigram tokenized .txt file. Remove usfm markings. Tokenize as trigrams with punction their own token. 
