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
	print("idf_description --- %s seconds ---" % (time.time()-start_time))
	start_time = time.time()
	idf_description = pickle.load(idf_description_handle)

with open('doc_norms_transcript.pickle', 'rb') as doc_norms_transcript_handle:
	print("doc_norms_transcript --- %s seconds ---" % (time.time()-start_time))
	start_time = time.time()
	doc_norms_transcript = pickle.load(doc_norms_transcript_handle)

with open('doc_norms_description.pickle', 'rb') as doc_norms_description_handle:
    print("doc_norms_description --- %s seconds ---" % (time.time()-start_time))
    doc_norms_description = pickle.load(doc_norms_description_handle)

with open('data/clus50K+clusterId_to_tedId2.pickle', 'rb') as clusterId_to_tedId_handle:
    print("clusterId_to_tedId2 --- %s seconds ---" % (time.time()-start_time))
    clusterId_to_tedId = pickle.load(clusterId_to_tedId_handle)

with open('data/clus50K+tedId_to_clusterId2.pickle', 'rb') as tedId_to_clusterId_handle:
    print("tedId_to_clusterId2 --- %s seconds ---" % (time.time()-start_time))
    tedId_to_clusterId = pickle.load(tedId_to_clusterId_handle)

print("svd_similarity --- %s seconds ---" % (time.time()-start_time))
svd_similarity = np.load("svd_similarity.pickle")

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


def get_docs_from_cluster(target_id, cluster, inv_idx, idf, svd_similarity, cluster_len):
    similarity_list = []
    for doc_id in cluster:
        if (doc_id != target_id):
            similarity_list.append((svd_similarity[target_id, doc_id], doc_id))
	top_docs = []
    # Subtract one to remove the target_id
    max_len = min(5, cluster_len - 1)
    while len(top_docs) < max_len:
        score, doc_id = max(similarity_list)
        top_docs.append(doc_id)
        similarity_list.remove((score, doc_id))
    return top_docs

@irsystem.route('/', methods=['GET'])
def search():
    query = request.args.get('search')
    data = []
    similar_talks = []
    cluster_res = []
    author_talks = []
    if query is None:
        output_message = ""
    elif not query:
        output_message = "Please enter a valid query"
    else:
        author_talks = search_by_author(query, all_talks)
        if len(author_talks) != 0:
            data = author_talks
        if len(data) < 10:
            top_10 = index_search(query, inv_idx_transcript, inv_idx_description, idf_transcript, idf_description, doc_norms_transcript, doc_norms_description)[:10]

            # Get cluster from top document
            top_talk_id = top_10[0][1]
            cluster_id = tedId_to_clusterId[top_talk_id]
            cluster_lst = clusterId_to_tedId[cluster_id]
            cluster_lst_len = len(cluster_lst)

            if cluster_lst_len > 1:
                top_cluster_talks = get_docs_from_cluster(top_talk_id, cluster_lst, inv_idx_transcript, idf_transcript, svd_similarity, cluster_lst_len)
                # May be the case that there is less than 5 docs in cluster
                for doc_id in top_cluster_talks:
                    if all_talks[doc_id] not in data and all_talks[doc_id] not in top_10:
                        cluster_res.append(all_talks[doc_id])

            for score, doc_id in top_10:
                if all_talks[doc_id] not in data and len(data) < 10:
                    data.append(all_talks[doc_id])
                    similar_talks.append(all_talks[doc_id])
                    num_additional += 1

            # User searches by authoer
            if len(author_talks) != 0:
                # Not enough results in cluster
                if (5 + len(author_talks) < len(cluster_res)):
                    sim_talks_add = 10 - len(author_talks) - len(cluster_res)
                    clus_talks_add = len(cluster_res)
                # Enough results in cluster
                else:
                    sim_talks_add = 10 - len(author_talks)
                    clus_talks_add = 10 - len(author_talks) - sim_talks_add
                data = author_talks + similar_talks[0:sim_talks_add] + cluster_res[0:clus_talks_add]
            # User searches by content
            else:
                sim_talks_add = 10 - len(cluster_res)
                data = similar_talks[0:sim_talks_add] + cluster_res

            if top_10[0][0] == 0:
                output_message = "No results for \"" + query + "\", but here are videos you may be interested in"
            else:
                output_message = "You searched for \"" + query + "\""

    return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=data, query=query)
