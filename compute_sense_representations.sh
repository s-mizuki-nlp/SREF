#!/bin/bash

python ./compute_sense_representations.py \
--input_path=./data/vectors/emb_glosses_aug_gloss+examples.txt \
--normalize_lemma_embeddings \
--inference_strategy="synset_then_lemma" \
--sense_level="lemma_key" \
--kappa=0.01 \
--nu_minus_dof=1.0 \
--cov=None
