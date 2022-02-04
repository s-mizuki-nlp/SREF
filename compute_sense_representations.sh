#!/bin/bash

python ./compute_sense_representations.py \
--input_path=./data/vectors/emb_glosses_aug_gloss+examples.txt \
--normalize_lemma_embeddings \
--inference_strategy="synset" \
--semantic_relation="all-relations" \
--kappa=0.01 \
--nu_minus_dof=0.01 \
--cov=10.0
