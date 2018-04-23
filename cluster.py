import ast
import numpy as np
import pandas as pd
import pickle

from collections import Counter
from collections import defaultdict
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

'''
Clustering Method
1. Features used
  - ratings: converted to one-hot vectors (rows 0-25) counts normalized
  - comments
  - duration
  - event: converted to one-hot vectors
  - film_date
  - languages
  - num_speakers
  - published_date
  - tags: converted to one-hot vectors of top 20 tags
  - views
2. PCA
'''
NUM_RATINGS_TYPES = 26
NUM_TAGS_TYPES = 20
CLUSTER_NUM = 50
DROP_ROWS = ['main_speaker', 'description', 'main_speaker', 'name', 'tags', 'ratings', 'related_talks', 'speaker_occupation', 'title', 'url']

'''
Rating Reference
{1: 'Beautiful',
 2: 'Confusing',
 3: 'Courageous',
 7: 'Funny',
 8: 'Informative',
 9: 'Ingenious',
 10: 'Inspiring',
 11: 'Longwinded',
 21: 'Unconvincing',
 22: 'Fascinating',
 23: 'Jaw-dropping',
 24: 'Persuasive',
 25: 'OK',
 26: 'Obnoxious'}
'''

def build_ratings_tags_dict(ratings_tags_dict, data):
  for talk_idx in range(len(data)):
    rating_lst = ast.literal_eval(data['ratings'][talk_idx])
    for rating in rating_lst:
      rating_id = rating['id']
      name = rating['name']
      ratings_tags_dict[rating_id] = name
      if len(ratings_tags_dict) == NUM_RATINGS_TYPES:
          break
  return ratings_tags_dict

def ratings_conversion(data):
  rows = data.shape[0]
  ratings_mtx = np.zeros((rows, NUM_RATINGS_TYPES))
  for talk_idx in range(len(data)):
    rating_lst = ast.literal_eval(data['ratings'][talk_idx])
    for rating in rating_lst:
      rating_id = rating['id'] - 1 # subtract for 0 index
      count = rating['count']
      ratings_mtx[talk_idx,rating_id] = count
  row_sum = np.sum(ratings_mtx, axis=1)
  ratings_mtx = ratings_mtx / row_sum.reshape(row_sum.size,1)
  # Data Clean
  valid_indices = np.sum(ratings_mtx, axis=0).nonzero()[0]
  ratings_mtx = ratings_mtx[:,valid_indices]
  df = pd.DataFrame(ratings_mtx, columns=list(ratings_tags_dict.values()))
  return df

def transformEventData(data):
  categorical = []
  for col, value in data.iteritems():
    if col == 'event':
      categorical.append(col)
  numerical = data.columns.difference(categorical)
  data_cat = data[categorical]
  data_cat = pd.get_dummies(data_cat)
  data_num = data[numerical]
  return pd.concat([data_num, data_cat], axis=1)

def build_tag_dict(data):
  tag_dict = defaultdict(int)
  for talk_idx in range(len(data)):
    tag_lst = ast.literal_eval(data['tags'][talk_idx])
    for tag in tag_lst:
      tag_dict[tag] += 1
  tag_count = Counter(tag_dict)
  return [tup[0] for tup in tag_count.most_common(NUM_TAGS_TYPES)]

def tags_conversion(data):
  rows = data.shape[0]
  tags_mtx = np.zeros((rows, NUM_TAGS_TYPES))
  for talk_idx in range(len(data)):
    tag_lst = ast.literal_eval(data['tags'][talk_idx])
    for tag_ind in range(len(top_tags)):
      top_tag = top_tags[tag_ind]
      if top_tag in tag_lst:
        tags_mtx[talk_idx,tag_ind] = 1
  df = pd.DataFrame(tags_mtx, columns=top_tags)
  return df

def build_cluster_mapping_and_size(cluster_labels):
    cnt_dict_size = defaultdict(int)
    cnt_dict = defaultdict(list)
    tedId_to_clusterId = {}
    for idx in range(len(cluster_labels)):
        clust_label = cluster_labels[idx]
        tedId_to_clusterId[idx] = clust_label
        cnt_dict_size[clust_label] += 1
        cnt_dict[clust_label].append(idx)
    return cnt_dict, cnt_dict_size, tedId_to_clusterId


if __name__ == "__main__":
  tedData = pd.read_csv('ted_main.csv')
  ratings_tags_dict = build_ratings_tags_dict({}, tedData)
  ratings_mtx = ratings_conversion(tedData)
  trans_tedData = transformEventData(tedData).drop(DROP_ROWS, axis=1)
  top_tags = build_tag_dict(tedData)
  cleaned_tedData = pd.concat([tags_data, trans_tedData,ratings_mtx], axis=1)
  kmeans = KMeans(init='k-means++', n_clusters=CLUSTER_NUM, n_init=CLUSTER_NUM)
  kmeans.fit(cleaned_tedData)
  kmeanlabels=kmeans.labels_
  clusterId_to_tedId, cnt_dict_size, tedId_to_clusterId = build_cluster_mapping_and_size(kmeanlabels)
  pickle.dump(clusterId_to_tedId, open( "data/clus50K+clusterId_to_tedId2.pickle", "wb" ) )
  pickle.dump(tedId_to_clusterId, open( "data/clus50K+tedId_to_clusterId2.pickle", "wb" ) )
