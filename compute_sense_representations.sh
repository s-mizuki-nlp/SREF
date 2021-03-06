#!/bin/bash

# MultiVariateNormal
: '
python ./compute_sense_representations.py \
--input_path=./data/vectors/emb_glosses_aug_gloss+examples.txt \
--prob_distribution="MultiVariateNormal" \
--normalize_lemma_embeddings \
--inference_strategy="synset-and-lemma" \
--semantic_relation="all-relations-but-hyponymy" \
--posterior_inference_method="mean_posterior" \
--posterior_inference_parameter_estimation="mean" \
--kappa=1.0 \
--nu_minus_dof=1.0 \
--cov=-1.0
'

# vonMisesFisher
python ./compute_sense_representations.py \
--input_path=./data/vectors/emb_glosses_aug_gloss+examples.txt \
--prob_distribution="vonMisesFisher" \
--inference_strategy="synset-and-lemma" \
--semantic_relation="all-relations" \
--prior_inference_method="independent" \
--posterior_inference_method="default" \
--posterior_inference_parameter_estimation="map" \
--whitening='{"pre_norm":False, "post_norm":True, "n_dim_reduced":128}' \
--c=10.0 \
--r_0=1.0
# --normalize_lemma_embeddings \