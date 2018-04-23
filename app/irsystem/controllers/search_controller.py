from . import *  
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
import csv
import math
import sys  
from collections import defaultdict, Counter
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import PorterStemmer
import ast

##
reload(sys)  
sys.setdefaultencoding('utf8')

project_name = "RecommenTED"
net_id = "Priyanka Rathnam: pcr43, Minzhi Wang: mw787, Emily Sun: eys27, Lillyan Pan: ldp54, Rachel Kwak sk2472"

tokenizer = TreebankWordTokenizer()
all_talks = {}

stemmer=PorterStemmer()

with open('new_transcripts.csv', 'rb') as transcript_file:
    transcript_reader = csv.reader(transcript_file)
    transcript_url_dict = dict(transcript_reader)

with open('new_descriptions.csv', 'rb') as description_file:
    description_reader = csv.reader(description_file)
    description_url_dict = dict(description_reader)

with open('ted_talks_ratings.csv') as csvfile:
	reader = csv.DictReader(csvfile)
	i = 0
	for row in reader:
		# ratings = ast.literal_eval(row['ratings'][1:][:-1])
		# #list of name, count tuples
		# name_count_list = [(rating["name"], rating["count"]) for rating in ratings]
		# rating_names = []
		# rating_counts = []
		# for rating in sorted(name_count_list):
		# 	rating_names.append(rating[0])
		# 	rating_counts.append(rating[1])

		all_talks[i] = {"title": row['title'], 
					   "description": (row["description"], "" if row['url'] not in description_url_dict else description_url_dict[row['url']]),
					   "speaker": row['main_speaker'], 
					   "tags": [word.strip('\'').strip(" ").strip('\'') for word in row["tags"][1:][:-1].split(",")], 
					   "url": row["url"], 
                       "transcript": "" if row['url'] not in transcript_url_dict else transcript_url_dict[row['url']],
					   "views": row['views'],
					   #"rating_names": rating_names,
					   #"rating_counts": rating_counts,
                       # Ratings
                       "Beautiful": row['Beautiful'],
                       "Confusing": row['Confusing'],
                       "Courageous": row['Courageous'],
                       "Funny": row['Funny'],
                       "Informative": row['Informative'],
                       "Ingenious": row['Ingenious'],
                       "Inspiring": row['Inspiring'],
                       "Longwinded": row['Longwinded'],
                       "Unconvincing": row['Unconvincing'],
                       "Fascinating": row['Fascinating'],
                       "Jawdropping": row['Jaw-dropping'],
                       "Persuasive": row['Persuasive'],
                       "OK": row['OK'],
                       "Obnoxious": row['Obnoxious']
                       }
		i += 1

def build_inverted_index(msgs, text_data_type):
    index = defaultdict(list)
    
    for i in range(0, len(msgs)):
        
        # Counter to count all occurences of word in tokenized message
        if text_data_type == "description":
            if msgs[i][text_data_type][1] == "":
                text_data = msgs[i][text_data_type][0]
            else:
                text_data = msgs[i][text_data_type][1]
        else:
            text_data = msgs[i][text_data_type]
        stemmed_counts = Counter(tokenizer.tokenize(text_data.lower()))
        
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
        # print(word)
        # print(type(idx))
        for doc_id, tf in idx:
            # Make sure word has not been pruned
            if word in idf:
                norms[doc_id] += (tf*idf[word])** 2
        
    return np.sqrt(norms)

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
    vocabulary = []

    for word_id in range(len(list_inv_index)):
        word, postings = list_inv_index[word_id]
        vocabulary.append(word)
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

svd_similarity = svd(inv_idx, idf)

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

inv_idx_transcript = build_inverted_index(all_talks, "transcript")
# print(inv_idx_transcript)
inv_idx_description = build_inverted_index(all_talks, "description")
# print(inv_idx_description)
idf_transcript = compute_idf(inv_idx_transcript, len(all_talks))
idf_description = compute_idf(inv_idx_description, len(all_talks))

# prune the terms left out by idf
inv_idx_transcript = {key: val for key, val in inv_idx_transcript.items() if key in idf_transcript}
inv_idx_description = {key: val for key, val in inv_idx_description.items() if key in idf_description}

doc_norms_transcript = compute_doc_norms(inv_idx_transcript, idf_transcript, len(all_talks))
doc_norms_description = compute_doc_norms(inv_idx_description, idf_description, len(all_talks))

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

            #this is a hacky solution: I always prepare 5 extra search results
            top_5 = index_search(query, inv_idx_transcript, inv_idx_description, idf_transcript, idf_description, doc_norms_transcript, doc_norms_description)[:5]
            for score, doc_id in top_5:
                if all_talks[doc_id] not in data and len(data) < 5:
                    data.append(all_talks[doc_id])

        output_message = "You searched for \"" + query + "\""
    print(data)
    return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=data, query=query)
