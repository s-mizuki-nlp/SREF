#!/bin/bash

: '
python evaluate_wsd_task_using_kNN.py \
--bert_host="musa:5555" \
--sense_representation_path="./data/representations/vonMisesFisher/norm-False_whitening_dim-None_str-synset-and-lemma_semrel-all-relations_prior-None_posterior-None_estimator-mle_c-nan_r0-nan_aug_gloss+examples.pkl" \
--sense_representation_type="vonMisesFisher" \
--similarity_metric="likelihood"
# --force_cosine_similarity_for_adj_and_adv
# --try-again
'

echo "=== No. 52 ==="
python evaluate_wsd_task_using_kNN.py \
--bert_host="musa:5555" \
--sense_representation_path="./data/representations/vonMisesFisher/norm-False_whitening_dim-128_str-synset-and-lemma_semrel-all-relations_prior-inherit_posterior-default_estimator-map_c-10.0_r0-1.0_aug_gloss+examples.pkl" \
--sense_representation_type="vonMisesFisher" \
--similarity_metric="cosine"

echo "=== No. 53 ==="
python evaluate_wsd_task_using_kNN.py \
--bert_host="musa:5555" \
--sense_representation_path="./data/representations/vonMisesFisher/norm-False_whitening_dim-128_str-synset-and-lemma_semrel-all-relations_prior-inherit_posterior-default_estimator-map_c-10.0_r0-1.0_aug_gloss+examples.pkl" \
--sense_representation_type="vonMisesFisher" \
--similarity_metric="likelihood"

echo "=== No. 54 ==="
python evaluate_wsd_task_using_kNN.py \
--bert_host="musa:5555" \
--sense_representation_path="./data/representations/vonMisesFisher/norm-False_whitening_dim-128_str-synset-and-lemma_semrel-all-relations_prior-independent_posterior-default_estimator-map_c-nan_r0-1.0_aug_gloss+examples.pkl" \
--sense_representation_type="vonMisesFisher" \
--similarity_metric="cosine"

echo "=== No. 55 ==="
python evaluate_wsd_task_using_kNN.py \
--bert_host="musa:5555" \
--sense_representation_path="./data/representations/vonMisesFisher/norm-False_whitening_dim-128_str-synset-and-lemma_semrel-all-relations_prior-independent_posterior-default_estimator-map_c-nan_r0-1.0_aug_gloss+examples.pkl" \
--sense_representation_type="vonMisesFisher" \
--similarity_metric="likelihood"
