import csv
import math
import sys  
from collections import defaultdict, Counter
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import PorterStemmer
import numpy as np
import ast
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

##
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
        ratings = ast.literal_eval(row['ratings'][1:][:-1])
        #list of name, count tuples
        name_count_list = [(rating["name"], rating["count"]) for rating in ratings]
        rating_names = []
        rating_counts = []
        for rating in sorted(name_count_list):
            rating_names.append(rating[0])
            rating_counts.append(rating[1])
        all_talks[i] = {"title": row['title'], 
                       "description": row['description'],
                       "speaker": row['main_speaker'], 
                       "tags": [word.strip('\'').strip(" ").strip('\'') for word in row["tags"][1:][:-1].split(",")], 
                       "url": row["url"], 
                       "views": row['views'],
                       "rating_names": rating_names,
                       "rating_counts": rating_counts}
        i += 1

def build_inverted_index(msgs):
    index = defaultdict(list)
    
    for i in range(0, len(msgs)):
        
        # Counter to count all occurences of word in tokenized message
        description = msgs[i]['description']
        stemmed_counts = Counter([stemmer.stem(word.decode('utf-8')) for word in tokenizer.tokenize(description.lower())])
        
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
    q = [stemmer.stem(word.decode('utf-8')) for word in tokenizer.tokenize(query.lower())]
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
talk_titles = []
for talk_id, talk in all_talks.items():
    talk_titles.append(talk["title"])

def svd_decomposition(inv_idx, idf):
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
    red_lsa = Normalizer(copy=False).fit_transform(red_lsa)
    similarity = np.asarray(np.asmatrix(red_lsa) * np.asmatrix(red_lsa).T)
    print(similarity)
    print(similarity.diagonal())
    #(file_vectors, weights, word_vectors) = np.linalg.svd(doc_word_counts, full_matrices=False)
    # word_vectors = word_vectors.T
    # print(file_vectors.shape)
    # print(weights.shape)
    # print(word_vectors.shape)
    # print(doc_word_counts[0])
    # with open("file_vectors.tsv", "w") as out:
    #     for i in range(len(file_vectors[0,:])):
    #         out.write("V{}\t".format(i))
    #     out.write("Title\n")
        
    #     for talk_id in range(len(talk_titles)):
    #         for i in range(len(file_vectors[talk_id,:])):
    #             out.write("{:.6f}\t".format(file_vectors[talk_id,i]))
    #         out.write("{}\n".format(all_talks[talk_id]))

    # with open("word_vectors.tsv", "w") as out:
    #     for i in range(len(word_vectors[0,:])):
    #         out.write("V{}\t".format(i))
    #     out.write("Word\n")
        
    #     for word_id in range(len(vocabulary)):
    #         for i in range(len(word_vectors[word_id,:])):
    #             out.write("{:.6f}\t".format(word_vectors[word_id,i]))
    #         out.write("{}\n".format(vocabulary[word_id]))
    
    return red_lsa

def sort_vector(v, names):
    sorted_list = sorted(list(zip(v, names)))
    for pair in sorted_list[:10]:
        print(pair)
    for pair in sorted_list[-10:]:
        print(pair)

