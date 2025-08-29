bitext-preprocessor.py is designed to take in NTs in 3 different languages. The input files need to be usfm titled 41.usfm-67.usfm. 
The output is a 3-language "bi-text", a tritext? A series of CSV files (41-47) with index locations.
Index locations are defined by biblical 8 digit. Matthew 2:3 is 41002003. First two digits identify the book, the next three the chapter, and the final three the verse. 

usfm_to_clean_txt.py is intended to take files 41.usfm to 67.usfm and convert them to one line for each verse in a .txt file. 
Cleaning involves removing everything except letters and white spaces. Converted to all lowercase as well. 

usfm_to_trigram.py is intended to take a usfm file and convert it to a trigram tokenized .txt file. Remove usfm markings. Tokenize as trigrams with punction their own token. 

alignment_scorer_nt_3gram.py is intended to take files 41.txt to 67.txt from subfolders source & target. These files are aligned by the line of txt file. Each text file is trigram tokenized 
and preprocessed text. The output is an alignment score comparing the source/41.txt to target/41.txt.  

alignment_scorer_plus_back_translation.py is intended to take files 41.txt to 67.txt from source & target as well as "back." These files are aligned by the line of txt file. Each text file is 
trigram tokenized and preprocessed text. The output is an alignment score comparing the source/41.txt to target/41.txt, assuming back/41.txt is a back translation of the target. 
The wrinkle added is that another high resource language text is added to act as a back translation of the target text. It does not need to be an actuakl back translation. For example,
for a target language (aka a zero- resource language) in Kenya may have a source of English. Then another high quality text is treated as if it were a back 
translation (e.g. a parallel Swahili text). 

alignment_scorer_neural.py is intended to adapt the alignment_scorer_plus_back_translation.py to a neural approach instead of a statistical one. 
