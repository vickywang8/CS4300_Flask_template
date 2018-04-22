from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import PorterStemmer

import csv
import sys  
import pickle

reload(sys)  
sys.setdefaultencoding('utf8')

tokenizer = TreebankWordTokenizer()

stemmer=PorterStemmer()

url_transcripts_dict = {}

with open('transcripts.csv') as transcript_file:
    transcript_reader = csv.DictReader(transcript_file)
    new_transcripts = []
    urls = []
    for t_row in transcript_reader:
    	new_transcript = ""
    	transcript = t_row["transcript"]
    	for word in tokenizer.tokenize(transcript.lower()):
    		try:
        		new_transcript += stemmer.stem(word.decode('utf-8')) + " "
        	except:
        		continue
        url_transcripts_dict[t_row["url"]] = new_transcript

with open("new_transcripts.pickle", "wb") as handle:
    pickle.dump(url_transcripts_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)