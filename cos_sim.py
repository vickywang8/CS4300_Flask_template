import numpy as np
import math
import csv
import ast
from collections import defaultdict, Counter
from nltk.tokenize import TreebankWordTokenizer
# from nltk.stem.wordnet import WordNetLemmatizer

# # attempt to expand query with lemmatization
# def query_expansion(msg):
# 	print msg
# 	new_msg = msg
# 	for word in msg:
# 		new = lemmatizer.lemmatize(word)
# 		if new != word:
# 			new_msg.append(new)
# 	print new_msg
# 	return new_msg

# JACCARD 
def build_jac(n_talks, all_talks, top_docs):
    jaccard = np.zeros((n_talks, n_talks))
    
    for i in range(0, n_talks):
        for j in range(i, n_talks):
            if (i==j):
                jaccard[i, j] = 1
                jaccard[j, i] = 1
                continue

            doc_i = top_docs[i]
            doc_j = top_docs[j]
            i_tags = set(ast.literal_eval(all_talks[doc_i]['tags']))
            j_tags = set(ast.literal_eval(all_talks[doc_j]['tags']))

            intersect = len(i_tags & j_tags)
            union = len(i_tags | j_tags)
            
            sim = float(intersect)/float(union)
                   
            jaccard[i, j] = sim
            jaccard[j, i] = sim

    return jaccard

def get_top_jac(top_id, jaccard, top_docs):
	mapping = {top_docs[i]:i for i in range(0, len(top_docs))}
	top_jaccard = jaccard[mapping[top_id]]
	sort_jac = np.argsort(np.array(top_jaccard))[::-1]

	return np.array(top_docs)[sort_jac]


def build_inverted_index(msgs):
    index = defaultdict(list)
    
    for i in range(0, len(msgs)):
        
        # Counter to count all occurences of word in tokenized message
        description = msgs[i]['description']
        counts = Counter(tokenizer.tokenize(description.lower()))
        
        # Add to dictionary
        for word in counts:
            index[word].append((i, counts[word]))
            
    return index


def compute_idf(inv_idx, n_docs, min_df=10, max_df_ratio=0.95):
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
    q = tokenizer.tokenize(query.lower())
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

if __name__ == "__main__":
	tokenizer = TreebankWordTokenizer()
	# lemmatizer = WordNetLemmatizer()

	all_talks = {}
	with open('ted_main.csv') as csvfile:
		reader = csv.DictReader(csvfile)
		i = 0
		for row in reader:
			all_talks[i] = {"title": row['title'], 
						   "description": row['description'], 
						   "speaker": row['main_speaker'], 
						   "ratings": row['ratings'],
						   "related_talks": row['related_talks'],
						   "tags": row["tags"], 
						   "url": row["url"], 
						   "views": row['views']}
			i += 1

	inv_idx = build_inverted_index(all_talks)
	idf = compute_idf(inv_idx, len(all_talks), min_df=10,  max_df_ratio=0.95)

	# prune the terms left out by idf
	inv_idx = {key: val for key, val in inv_idx.items() if key in idf}

	doc_norms = compute_doc_norms(inv_idx, idf, len(all_talks))

	sample_query = "facebook"
	top_10 = index_search(sample_query, inv_idx, idf, doc_norms)[:10]

	top_docs = [doc_id for _, doc_id in top_10]

	top_id = top_10[0][1]

	jac = build_jac(len(top_docs), all_talks, top_docs)

	top_jac = get_top_jac(top_id, jac, top_docs)


