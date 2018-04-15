import numpy as np
import math
import csv
from collections import defaultdict
from nltk.tokenize import TreebankWordTokenizer

ted_dict = {}

with open('ted_main.csv') as csvfile:
	reader = csv.DictReader(csvfile)
	i = 0
	for row in reader:
		ted_dict[i] = {"title": row['title'], 
					   "description": row['description'], 
					   "speaker": row['main_speaker'], 
					   "tags": row["tags"], 
					   "url": row["url"], 
					   "views": row['views']}
		i += 1

tokenizer = TreebankWordTokenizer()
tokenize_blurb = tokenizer.tokenize(blurb)

