import csv
import math
import sys  
from collections import defaultdict, Counter
from nltk.tokenize import TreebankWordTokenizer
import datetime
import pickle
import numpy as np
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import NMF
import scipy.sparse
from sklearn.pipeline import Pipeline

##
reload(sys)  
sys.setdefaultencoding('utf8')

tokenizer = TreebankWordTokenizer()
all_talks = {}
month_num_to_abbr = {"1": "Jan", "2": "Feb", "3": "Mar", "4": "Apr", "5": "May", "6": "Jun", "7": "Jul", "8": "Aug", "9": "Sept", "10": "Oct", "11":"Nov", "12": "Dec"}

with open('new_transcripts.pickle', 'rb') as transcript_handle:
    transcript_url_dict = pickle.load(transcript_handle)

with open('new_descriptions.pickle', 'rb') as description_handle:
    description_url_dict = pickle.load(description_handle)

with open('ted_talks_ratings.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    i = 0
    for row in reader:
        talk_time = datetime.datetime.fromtimestamp(int(row['published_date']))
        all_talks[i] = {"title": row['title'], 
                       "description": (row["description"], "" if row['url'] not in description_url_dict else description_url_dict[row['url']]),
                       "speaker": row['main_speaker'], 
                       "tags": [word.strip('\'').strip(" ").strip('\'') for word in row["tags"][1:][:-1].split(",")], 
                       "url": row["url"], 
                       "transcript": "" if row['url'] not in transcript_url_dict else transcript_url_dict[row['url']],
                       "views": row['views'],
                       "year": month_num_to_abbr[str(talk_time.month)] + " " + str(talk_time.year),
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

def get_tfidf(inv_idx, idf):
  doc_word_counts = np.zeros([ len(all_talks), len(inv_idx) ])
  list_inv_index = list(inv_idx.items())
  for word_id in range(len(list_inv_index)):
    word, postings = list_inv_index[word_id]
    for d_id, tf in postings:
      doc_word_counts[d_id, word_id] = tf*idf[word]
  return doc_word_counts

def svd(tfidf):
  # modified from http://www.datascienceassn.org/sites/default/files/users/user1/lsa_presentation_final.pdf
  lsa = TruncatedSVD(200, algorithm = 'randomized')
  red_lsa = lsa.fit_transform(tfidf)
  #print(red_lsa)
  red_lsa = Normalizer(copy=False).fit_transform(red_lsa)
  #print(red_lsa)
  similarity = np.asarray(np.asmatrix(red_lsa) * np.asmatrix(red_lsa).T)
  #print(similarity)
  #print(similarity.diagonal())
  return similarity

##code modified from https://www.kaggle.com/adelsondias/ted-talks-topic-models/notebook
def topic_modeling(tfidf, idx):
  num_topics = 100
  nmf = NMF(n_components=num_topics,random_state=0)
  topics = nmf.fit_transform(tfidf)
  topic_dict = {}
  doc_topic_score = np.zeros([ len(tfidf), num_topics ])
  for topic_id, lst in enumerate(nmf.components_):
    top_ten = lst.argsort()[-10:]
    top_ten = top_ten[::-1]
    words = [idx[i] for i in top_ten]
    topic_dict[topic_id] = words
    for doc_id in all_talks.keys():
      score = 0
      for word_id in top_ten:
        score+=tfidf[doc_id,word_id]
      doc_topic_score[doc_id, topic_id] = score
  return topic_dict, doc_topic_score

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

tfidf = get_tfidf(inv_idx_transcript, idf_transcript)
idx_transcript = {i: t for i, t in enumerate(inv_idx_transcript)}

svd_similarity = svd(tfidf)
#svd_similarity = scipy.sparse.csc_matrix(svd_similarity)
# np.savetxt("svd_similarity1.txt", svd_similarity[:,:1000], delimiter=',')
# np.savetxt("svd_similarity2.txt", svd_similarity[:,-(len(svd_similarity[0])-1000):], delimiter=',')
#scipy.sparse.save_npz('sparse_matrix.npz', svd_similarity)
np.save("svd_similarity", svd_similarity)

topic_dict, doc_topic_score = topic_modeling(tfidf, idx_transcript)
np.save("doc_topic_score", doc_topic_score)

topic_name_dict = {0: "Self Reflection", 2: "Ants", 3: "Biology", 4: "Climate Change", 5: "Brain", 6: "Finance", 7: "Robots", 8: "Computer Security", 9: "Outer Space", 10: "Spiders", 11: "Mars", 12: "Galaxies", 13: "Viruses", 14: "Insects", 15: "Genetics", 16: "Water", 18: "Nuclear", 19: "Sea Life", 20: "Black Holes", 21: "Urban", 22: "Cancer", 23: "Gender", 24: "Coral Reefs", 26: "China", 29: "Refugees", 31: "Music", 32: "Gaming", 33: "War", 34: "Politics", 36: "Middle East", 37: "Africa", 38: "Sex Trade", 40: "Computers", 41: "Injuries", 42: "Geometry", 43: "Microbiology", 45: "Socioeconomic", 46: "Bacteria", 47: "Malaria", 48: "Fish", 50: "Disease", 51: "Food", 53: "Outdoors", 54: "Autism", 55: "India", 56: "Fireflies", 57: "Dinosaurs", 58: "Data", 59: "LGBTQ", 60: "Beetles", 61: "Donald Trump", 63: "String Theory", 64: "Addiction", 65: "Bats", 66: "Morality", 67: "Drugs", 68: "Prosthetics", 69: "Oil", 70: 'Quantum Mechanics', 71: 'Breast Cancer', 72: 'Particle Physics', 73: 'Art', 74: 'Hormones', 75: 'National Security', 76: 'Sleep', 77: 'Genes', 78: 'Forest', 79: 'Energy', 80: 'Healthcare', 81: 'Crime', 82: 'Patent', 83: 'Consciousness', 84: 'Stress', 85: 'Ebola', 86: 'Education', 87: 'Family', 88: 'Agriculture', 89: 'Laser', 90: 'Iran', 91: 'Banking', 92: 'Arctic', 93: 'Religion', 94: 'Antibiotics', 95: 'Language', 98: 'Car Accident', 99: 'Feminism'}

name_topic_dict = {v: k for k, v in topic_name_dict.iteritems()}

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

with open("topic_dict.pickle", "wb") as handle:
    pickle.dump(topic_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("topic_name_dict.pickle", "wb") as handle:
    pickle.dump(topic_name_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("name_topic_dict.pickle", "wb") as handle:
    pickle.dump(name_topic_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)



