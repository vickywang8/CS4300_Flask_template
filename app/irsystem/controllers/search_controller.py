from . import *  
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
import math
import sys  
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import PorterStemmer
import ast
import pickle
import time
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer

##
reload(sys)  
sys.setdefaultencoding('utf8')

project_name = "RecommenTED"
net_id = "Priyanka Rathnam: pcr43, Minzhi Wang: mw787, Emily Sun: eys27, Lillyan Pan: ldp54, Rachel Kwak sk2472"

tokenizer = TreebankWordTokenizer()

stemmer=PorterStemmer()

start_time = time.time()

with open('new_transcripts.pickle', 'rb') as transcript_handle:
	print("new_transcripts.pickle --- %s seconds ---" % (time.time()-start_time))
	start_time = time.time()
	transcript_url_dict = pickle.load(transcript_handle)

with open('new_descriptions.pickle', 'rb') as description_handle:
	print("new_descriptions --- %s seconds ---" % (time.time()-start_time))
	start_time = time.time()
	description_url_dict = pickle.load(description_handle)

with open('all_talks.pickle') as all_talks_handle:
	print("all_talks --- %s seconds ---" % (time.time()-start_time))
	start_time = time.time()
	all_talks = pickle.load(all_talks_handle)

with open('inv_idx_transcript.pickle', 'rb') as inv_transcript_handle:
	print("inv_idx_transcript --- %s seconds ---" % (time.time()-start_time))
	start_time = time.time()
	inv_idx_transcript = pickle.load(inv_transcript_handle)

with open('inv_idx_description.pickle', 'rb') as inv_description_handle:
	print("inv_idx_description --- %s seconds ---" % (time.time()-start_time))
	start_time = time.time()
	inv_idx_description = pickle.load(inv_description_handle)

with open('idf_transcript.pickle', 'rb') as idf_transcript_handle:
	print("idf_transcript --- %s seconds ---" % (time.time()-start_time))
	start_time = time.time()
	idf_transcript = pickle.load(idf_transcript_handle)

with open('idf_description.pickle', 'rb') as idf_description_handle:
	print("--- %s seconds ---" % (time.time()-start_time))
	start_time = time.time()
	idf_description = pickle.load(idf_description_handle)

with open('doc_norms_transcript.pickle', 'rb') as doc_norms_transcript_handle:
	print("--- %s seconds ---" % (time.time()-start_time))
	start_time = time.time()
	doc_norms_transcript = pickle.load(doc_norms_transcript_handle)

with open('doc_norms_description.pickle', 'rb') as doc_norms_description_handle:
    doc_norms_description = pickle.load(doc_norms_description_handle)

def compute_score(q, index, idf, doc_norms, q_weights):
    results = np.zeros(len(doc_norms))
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
    return results/(doc_norms*q_norm+1)

def index_search(query, transcript_index, description_index, transcript_idf, description_idf, transcript_doc_norms, description_doc_norms):     
    # Tokenize query
    q = [stemmer.stem(word.decode('utf-8')) for word in tokenizer.tokenize(query.lower())]
    q_weights = {}
    
    transcript_scores = compute_score(q, transcript_index, transcript_idf, transcript_doc_norms, q_weights)
    description_scores = compute_score(q, description_index, description_idf, description_doc_norms, q_weights)
            
   # change results to (score, doc_id) format   
    results = [(transcript_scores[i] + description_scores[i], i) for i in range(0, len(transcript_scores))]
    
    # sort results by score
    results.sort()

    return results[::-1]

def search_by_author(name, all_talks):
    talks_by_author = []
    for key, value in all_talks.items():
        if value["speaker"].lower() == name.lower():
            talks_by_author.append(value)
    return talks_by_author

def svd(inv_idx, idf):
	doc_word_counts = np.zeros([ len(all_talks), len(inv_idx) ])
	list_inv_index = list(inv_idx.items())
	for word_id in range(len(list_inv_index)):
		word, postings = list_inv_index[word_id]
		for d_id, tf in postings:
			doc_word_counts[d_id, word_id] = tf*idf[word]
	# modified from http://www.datascienceassn.org/sites/default/files/users/user1/lsa_presentation_final.pdf
	lsa = TruncatedSVD(200, algorithm = 'randomized')
	red_lsa = lsa.fit_transform(doc_word_counts)
	#print(red_lsa)
	red_lsa = Normalizer(copy=False).fit_transform(red_lsa)
	#print(red_lsa)
	similarity = np.asarray(np.asmatrix(red_lsa) * np.asmatrix(red_lsa).T)
	#print(similarity)
	#print(similarity.diagonal())
	return similarity

def get_docs_from_cluster(target_id, cluster, inv_idx, idf, svd_similarity):
	similarity_list = []
	for doc_id in cluster:
		similarity_list.append((svd_similarity[target_id, doc_id], doc_id))
	top_5 = []
	while len(top_5)<5:
		score, doc_id = max(similarity_list)
		top_5.append(doc_id)
		similarity_list.remove((score, doc_id))
	return top_5

svd_similarity = svd(inv_idx_transcript, idf_transcript)

@irsystem.route('/', methods=['GET'])
def search():
    query = request.args.get('search')
    data = []
    if query is None:
        output_message = ""
    elif not query:
        output_message = "Please enter a valid query"
    else:
        author_talks = search_by_author(query, all_talks)
        if len(author_talks) != 0:
            data = author_talks
        if len(data) < 5:
            num_additional = 5 - len(data)

            #this is a hacky solution: I always prepare 5 extra search results
            top_5 = index_search(query, inv_idx_transcript, inv_idx_description, idf_transcript, idf_description, doc_norms_transcript, doc_norms_description)[:5]
            for score, doc_id in top_5:
                if all_talks[doc_id] not in data and len(data) < 5:
                    data.append(all_talks[doc_id])
            print top_5[0][0]
            if top_5[0][0] == 0:
                output_message = "No results for \"" + query + "\", but here are videos you may be interested in"
            else:
                output_message = "You searched for \"" + query + "\""

        output_message = "You searched for \"" + query + "\""
    # print(data)
    return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=data, query=query)
