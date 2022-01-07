# knowledge-based. enable Try-Again, use augmented gloss
MODE=${1}

if [ "$MODE" = "original" ]; then
  EMB_PATH="./data/vectors/original/emb_wn.pkl"
elif [ "$MODE" = "ours" ]; then
  EMB_PATH="./data/vectors/emb_wn_all-relations_aug_gloss+examples.pkl"
fi

echo "PATH: ${EMB_PATH}"

python eval_nn.py \
-lemma_embeddings_path ${EMB_PATH} \
--sec_wsd \
-emb_strategy default \
-bert_host musa
