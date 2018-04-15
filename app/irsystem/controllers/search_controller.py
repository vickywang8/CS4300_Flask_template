from . import *  
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
import csv
import math
import sys  
from collections import defaultdict, Counter
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import PorterStemmer

reload(sys)  
sys.setdefaultencoding('utf8')

project_name = "RecommenTED"
net_id = "Priyanka Rathnam: pcr43, Minzhi Wang: mw787, Emily Sun: eys27, Lillyan Pan: ldp54, Rachel Kwak sk2472"

tokenizer = TreebankWordTokenizer()
all_talks = {}

stemmer=PorterStemmer()

with open('ted_main.csv') as csvfile:
	reader = csv.DictReader(csvfile)
	i = 0
	for row in reader:
		all_talks[i] = {"title": row['title'], 
					   "description": row['description'],
					   "speaker": row['main_speaker'], 
					   "tags": row["tags"], 
					   "url": row["url"], 
					   "views": row['views']}
		i += 1

# def getstems(sent):
#     return [stemmer.stem(w.lower()) for w in word_splitter.findall(sent)]

def build_inverted_index(msgs):
    index = defaultdict(list)
    
    for i in range(0, len(msgs)):
        
        # Counter to count all occurences of word in tokenized message
        description = msgs[i]['description']
        stemmed_counts = Counter([stemmer.stem(word) for word in tokenizer.tokenize(description.lower())])
        
        # Add to dictionary
        for word in stemmed_counts:
            index[word].append((i, stemmed_counts[word]))
            
    return index


def compute_idf(inv_idx, n_docs, min_df=1, max_df_ratio=0.80):
    idf = {}
    
    for word, idx in inv_idx.items():
        word_docs = len(idx)
        
        # Word in too few documents
        if word_docs < min_df:
            continue
        # Word in > 95% docs
        elif word_docs/n_docs > max_df_ratio:
            continue
        else:
            idf[word] = math.log(n_docs/(1+word_docs), 2)
    
    return idf


def compute_doc_norms(index, idf, n_docs):
    norms = np.zeros(n_docs)
    
    for word, idx in index.items():
        for doc_id, tf in idx:
            # Make sure word has not been pruned
            if word in idf:
                norms[doc_id] += (tf*idf[word])** 2
        
    return np.sqrt(norms)


def index_search(query, index, idf, doc_norms):     
    results = np.zeros(len(doc_norms))
    
    # Tokenize query
    q = [stemmer.stem(word) for word in tokenizer.tokenize(query.lower())]
    q_weights = {}
    
    for term in q:
        postings = []
        if term not in index.keys():
            continue
        else:
            postings = index[term]
            
        for doc_id, tf in postings:
            wij = tf*idf[term]
            wiq = q.count(term)*idf[term]
            q_weights[term] = wiq

            results[doc_id] += wij*wiq
    
    # Find query norm
    q_norm = 0
    for w in q_weights.values():
        q_norm += w*w
    q_norm = math.sqrt(q_norm)
    
    # Normalize
    results = results/(doc_norms*q_norm+1)
            
   # change results to (score, doc_id) format   
    results = [(results[i], i) for i in range(0, len(results))]
    
    # sort results by score
    results.sort()

    return results[::-1]

def search_by_author(name, all_talks):
    talks_by_author = []
    for key, value in all_talks.items():
        if value["speaker"].lower() == name.lower():
            talks_by_author.append(value)
    return talks_by_author

inv_idx = build_inverted_index(all_talks)
idf = compute_idf(inv_idx, len(all_talks))

# prune the terms left out by idf
inv_idx = {key: val for key, val in inv_idx.items() if key in idf}

doc_norms = compute_doc_norms(inv_idx, idf, len(all_talks))

@irsystem.route('/', methods=['GET'])
def search():
    query = request.args.get('search')
    data = []
    if not query:
	   output_message = ''
    else:
        author_talks = search_by_author(query, all_talks)
        if len(author_talks) != 0:
            data = author_talks
        if len(data) < 5:
            num_additional = 5 - len(data)
            top_n = index_search(query, inv_idx, idf, doc_norms)[:num_additional]
            for score, doc_id in top_n:
                data.append(all_talks[doc_id])

        output_message = "Your search: " + query
    return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=data)
