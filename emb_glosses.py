import time
import argparse
import logging
from collections import defaultdict

import numpy as np
from nltk.corpus import wordnet as wn
from nltk import word_tokenize
from tqdm import tqdm

from bert_as_service import BertEncoder
# from bert_as_service import bert_embed


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def wn_synset2keys(synset):
    if isinstance(synset, str):
        synset = wn.synset(synset)
    return list(set([lemma.key() for lemma in synset.lemmas()]))


def fix_lemma(lemma):
    return lemma.replace('_', ' ')


def get_sense_data(emb_strategy: str):
    data = []
    import pickle
    name = locals()
    for pos in ['n', 'r', 'v', 'a']:
        pos_name = f"{pos}_example"
        if emb_strategy.startswith("aug"):
            name[pos_name] = pickle.load(open(f'./sentence_dict_{pos}', 'rb'))
            name[pos_name] = {key: [v for v in value] for key, value in name[pos_name].items() if value}
            print(f'{pos} sentences loaded!: {len(name[pos_name])}')
        else:
            name[pos_name] = {}
    type2pos = {1: 'n', 2: 'v', 3: 'a', 4: 'r', 5: 'a'}
    for index, synset in enumerate(wn.all_synsets()):
        # all_lemmas = list of
        all_lemmas = [fix_lemma(lemma.name()) for lemma in synset.lemmas()]
        # gloss: definition sentence
        gloss = ' '.join(word_tokenize(synset.definition()))
        ty = int([i.key() for i in synset.lemmas()][0].split('%')[1][0])
        pos = type2pos[ty]
        pos_name = f"{pos}_example"
        synset_id = synset.name()
        if synset_id in name[pos_name]:
            # concat examples in augmented corpora -> ./sentence_dict_{pos}.
            # sentences are re-tokenized using word_tokenize() function.
            examples = ' '.join( word_tokenize(' '.join(name[pos_name][synset_id]) ) )
        else:
            # do not use examples in augmented corpora
            examples = ''
        if emb_strategy.endswith("examples"):
            # concat examples in WordNet gloss.
            # sentences are re-tokenized using word_tokenize() function.
            examples += ' '.join( word_tokenize(' '.join(synset.examples())) )
        # for each lemma; append all lemmas, definition sentence and example sentences (may include augmented corpora)
        for lemma in synset.lemmas():
            lemma_name = fix_lemma(lemma.name())
            d_str = lemma_name + ' - ' + ' , '.join(all_lemmas) + ' - ' + gloss + examples
            data.append((synset, lemma.key(), d_str))

    data = sorted(data, key=lambda x: x[0])
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates sense embeddings based on glosses and lemmas.')
    parser.add_argument("-bert_host", required=True, help="bert-as-service hostname and ip address. e.g. localhost:5555")
    parser.add_argument('-batch_size', type=int, default=64, help='Batch size (BERT)', required=False)
    parser.add_argument('-emb_strategy', type=str, default='aug_gloss',
                        choices=['gloss', 'gloss+examples', 'aug_gloss', 'aug_gloss+examples'],
                        help='different components to learn the basic sense embeddings', required=False)
    parser.add_argument('-out_path', help='Path to resulting vector set', required=False,
                        default='data/vectors/emb_glosses_%s.txt')
    args = parser.parse_args()
    pooling_strategy = 'REDUCE_MEAN' # important parameter to replicate results using bert-as-service

    logging.info(f"connecting to bert-as-service: {args.bert_host}")
    host_info = args.bert_host.split(":")
    if len(host_info) == 1:
        bert_encoder = BertEncoder(host=host_info[0])
    elif len(host_info) == 2:
        bert_encoder = BertEncoder(host=host_info[0], port=host_info[1])
    else:
        raise ValueError(f"unexpected `bert_host` value: {args.bert_host}")

    logging.info('Preparing Gloss Data ...')
    glosses = get_sense_data(args.emb_strategy)
    glosses_vecs = defaultdict(list)

    logging.info('Embedding Senses ...')
    t0 = time.time()
    for batch_idx, glosses_batch in enumerate(tqdm(chunks(glosses, args.batch_size))):
        dfns = [e[-1] for e in glosses_batch]

        dfns_bert = bert_encoder.bert_embed_sents(dfns)

        for (synset, sensekey, dfn), dfn_bert in zip(glosses_batch, dfns_bert):
            dfn_vec = dfn_bert[1]
            glosses_vecs[sensekey].append(dfn_vec)

        t_span = time.time() - t0
        n = (batch_idx + 1) * args.batch_size
        logging.info('%d/%d at %.3f per sec' % (n, len(glosses), n/t_span))

    logging.info('Writing Vectors %s ...' % args.out_path)
    import pickle
    pickle.dump(glosses_vecs, open(args.out_path % str(args.emb_strategy), 'wb'), -1)
