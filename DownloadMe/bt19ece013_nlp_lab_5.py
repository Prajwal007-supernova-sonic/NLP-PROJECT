# Importing libraries
import nltk
import numpy as np
import pandas as pd
import random
from nltk import TreebankWordTokenizer
#from sklearn.model_selection import train_test_split
import pprint, time
 
#download the treebank corpus from nltk
#nltk.download('treebank')
 
#download the universal tagset from nltk
#nltk.download('universal_tagset')
 
# reading the Treebank tagged sentences
nltk_data = list(nltk.corpus.treebank.tagged_sents(tagset='universal'))
 
#print the first two sentences along with tags
print(nltk_data[:2])
