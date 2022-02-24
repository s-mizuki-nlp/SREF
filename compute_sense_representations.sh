#!/bin/bash

python ./compute_sense_representations.py \
--input_path=./data/vectors/emb_glosses_aug_gloss+examples.txt \
--normalize_lemma_embeddings \
--inference_strategy="synset-and-lemma" \
--semantic_relation="all-relations" \
--posterior_inference_method="mean_posterior" \
--posterior_inference_parameter_estimation="mean" \
--kappa=0.1 \
--nu_minus_dof=0.1 \
--cov=-1.0
