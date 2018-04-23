import csv
import math
import sys  
from collections import defaultdict, Counter
from nltk.tokenize import TreebankWordTokenizer
import ast
import pickle
import numpy as np
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer

##
reload(sys)  
sys.setdefaultencoding('utf8')

tokenizer = TreebankWordTokenizer()
all_talks = {}

with open('new_transcripts.pickle', 'rb') as transcript_handle:
    transcript_url_dict = pickle.load(transcript_handle)

with open('new_descriptions.pickle', 'rb') as description_handle:
    description_url_dict = pickle.load(description_handle)

with open('ted_talks_ratings.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    i = 0
    for row in reader:

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


def compute_idf(inv_idx, n_docs, min_df=10, max_df_ratio=0.80):
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

svd_similarity = svd(inv_idx_transcript, idf_transcript)
svd_similarity.dump("svd_similarity.pickle")

with open("all_talks.pickle", "wb") as handle:
    pickle.dump(all_talks, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("inv_idx_transcript.pickle", "wb") as handle:
    pickle.dump(inv_idx_transcript, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("inv_idx_description.pickle", "wb") as handle:
    pickle.dump(inv_idx_description, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("idf_transcript.pickle", "wb") as handle:
    pickle.dump(idf_transcript, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("idf_description.pickle", "wb") as handle:
    pickle.dump(idf_description, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("doc_norms_transcript.pickle", "wb") as handle:
    pickle.dump(doc_norms_transcript, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("doc_norms_description.pickle", "wb") as handle:
    pickle.dump(doc_norms_description, handle, protocol=pickle.HIGHEST_PROTOCOL)



