from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import PorterStemmer

import csv
import sys  

reload(sys)  
sys.setdefaultencoding('utf8')

tokenizer = TreebankWordTokenizer()

stemmer=PorterStemmer()

url_description_dict = {}

with open('ted_main.csv') as csvfile:
	reader = csv.DictReader(csvfile)
	for row in reader:
		description = row['description']
		new_description = ""
		for word in tokenizer.tokenize(description.lower()):
			try:
				new_description += stemmer.stem(word.decode('utf-8')) + " "
			except:
				new_description += word
		url_description_dict[row["url"]] = new_description

with open("new_descriptions.csv", "w") as resultFile:
    writer = csv.writer(resultFile)
    for key, value in url_description_dict.items():
    	writer.writerow([key, value])