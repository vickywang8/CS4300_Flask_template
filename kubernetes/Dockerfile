# Read from Ubuntu Base Image
FROM python:2.7
RUN mkdir -p /service
# Copy over all the files of interest
ADD app /service/app
ADD app.py /service/app.py
ADD config.py /service/config.py
ADD manage.py /service/manage.py
ADD requirements.txt /service/requirements.txt
ADD new_transcripts.pickle /service/new_transcripts.pickle
ADD new_descriptions.pickle /service/new_descriptions.pickle
ADD all_talks.pickle /service/all_talks.pickle
ADD inv_idx_transcript.pickle /service/inv_idx_transcript.pickle
ADD inv_idx_description.pickle /service/inv_idx_description.pickle
ADD idf_transcript.pickle /service/idf_transcript.pickle
ADD idf_description.pickle /service/idf_description.pickle
ADD doc_norms_transcript.pickle /service/doc_norms_transcript.pickle
ADD doc_norms_description.pickle /service/doc_norms_description.pickle
ADD 100scclusterId_to_tedId2.pickle /service/100scclusterId_to_tedId2.pickle
ADD 100sctedId_to_clusterId.pickle /service/100sctedId_to_clusterId.pickle
ADD svd_similarity.npy /service/svd_similarity.npy
ADD doc_topic_score.npy /service/doc_topic_score.npy
ADD topic_dict.pickle /service/topic_dict.pickle
ADD topic_name_dict.pickle /service/topic_name_dict.pickle
ADD name_topic_dict.pickle /service/name_topic_dict.pickle
WORKDIR /service/
RUN pip install -r requirements.txt
CMD python -u app.py $APP_SETTINGS $DATABASE_URL
